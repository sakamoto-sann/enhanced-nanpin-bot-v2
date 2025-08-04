#!/usr/bin/env python3
"""
🎲 Optimized Monte Carlo Risk Analysis for Nanpin Strategy
Enhanced performance and trade frequency optimization
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import concurrent.futures
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class OptimizedMonteCarloRiskAnalyzer:
    def __init__(self, start_date="2020-01-01", end_date="2024-12-31"):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.simulations = 1500  # Balanced for accuracy and speed
        
        # BALANCED Strategy parameters for optimal trade frequency
        self.strategy_params = {
            'min_drawdown': -15,  # Balanced (was -18, then -12)
            'max_fear_greed': 40,  # Balanced (was 35, then 45)
            'min_days_since_ath': 5,  # Balanced (was 7, then 3)
            'fibonacci_levels': {
                '23.6%': {'ratio': 0.236, 'base_multiplier': 1.8, 'entry_window': (-10.0, -0.5)},
                '38.2%': {'ratio': 0.382, 'base_multiplier': 2.8, 'entry_window': (-15.0, -0.5)},
                '50.0%': {'ratio': 0.500, 'base_multiplier': 4.5, 'entry_window': (-18.0, -0.5)},
                '61.8%': {'ratio': 0.618, 'base_multiplier': 7.0, 'entry_window': (-22.0, -0.5)},
                '78.6%': {'ratio': 0.786, 'base_multiplier': 11.0, 'entry_window': (-25.0, -0.5)}
            },
            'base_leverage': 2.8,  # Balanced
            'max_leverage': 16.0,  # Balanced
            'cooldown_hours': 36,  # Balanced (was 48, then 24)
            'min_range_pct': 0.04,  # Balanced (was 0.05, then 0.03)
            'score_threshold': 4.0,  # Balanced (was 5.0, then 3.0)
            'max_position_pct': 0.18,  # Balanced
            'min_capital': 75  # Balanced
        }
        
        self.results = []
        self.risk_metrics = {}
        
        # Performance tracking
        self.cpu_count = mp.cpu_count()
        print(f"🚀 Optimized Monte Carlo Analyzer initialized")
        print(f"   CPU cores available: {self.cpu_count}")
        print(f"   Simulations: {self.simulations}")
        print(f"   Strategy: Balanced Enhanced Goldilocks Plus")
    
    async def run_monte_carlo_analysis(self):
        """Run optimized Monte Carlo analysis with parallel processing"""
        print(f"🎲 OPTIMIZED MONTE CARLO RISK ANALYSIS")
        print("=" * 65)
        print(f"Simulations: {self.simulations}")
        print(f"Strategy: Enhanced Goldilocks Plus Nanpin")
        print(f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        
        # Load historical data
        await self._load_data()
        
        # Run parallel simulations
        await self._run_parallel_simulations()
        
        # Calculate risk metrics
        self._calculate_risk_metrics()
        
        # Run stress tests
        self._stress_test_scenarios()
        
        # Display results
        self._display_results()
        
        # Create visualizations
        self._create_optimized_visualizations()
        
        return self.results, self.risk_metrics
    
    async def _load_data(self):
        """Load and preprocess Bitcoin data"""
        try:
            # Download BTC data
            ticker = yf.Ticker("BTC-USD")
            self.btc_data = ticker.history(
                start=self.start_date.strftime('%Y-%m-%d'),
                end=self.end_date.strftime('%Y-%m-%d'),
                interval="1d"
            )
            
            print(f"✅ Data loaded: {len(self.btc_data)} days")
            
            # Calculate returns and volatility
            self.daily_returns = self.btc_data['Close'].pct_change().dropna()
            self.return_mean = self.daily_returns.mean()
            self.return_std = self.daily_returns.std()
            
            print(f"   Historical volatility: {self.return_std * np.sqrt(365) * 100:.1f}%")
            print(f"   Mean daily return: {self.return_mean * 100:.3f}%")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    async def _run_parallel_simulations(self):
        """Run simulations in parallel for better performance"""
        print(f"\n🔄 Running {self.simulations} optimized Monte Carlo simulations...")
        
        # Split simulations across CPU cores
        chunk_size = max(1, self.simulations // self.cpu_count)
        simulation_chunks = [
            range(i, min(i + chunk_size, self.simulations))
            for i in range(0, self.simulations, chunk_size)
        ]
        
        # Run simulations in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.cpu_count) as executor:
            tasks = []
            for chunk in simulation_chunks:
                task = executor.submit(self._run_simulation_chunk, chunk)
                tasks.append(task)
            
            # Collect results
            chunk_results = []
            for i, future in enumerate(concurrent.futures.as_completed(tasks)):
                results = future.result()
                chunk_results.extend(results)
                if (i + 1) % max(1, len(tasks) // 10) == 0:
                    print(f"   Completed chunk {i + 1}/{len(tasks)}")
        
        self.results = pd.DataFrame(chunk_results)
        print(f"✅ Completed {len(self.results)} simulations")
    
    def _run_simulation_chunk(self, sim_range):
        """Run a chunk of simulations (for parallel processing)"""
        results = []
        for sim_num in sim_range:
            # Generate synthetic prices
            synthetic_prices = self._generate_optimized_synthetic_prices()
            
            # Run strategy simulation
            result = self._run_optimized_strategy_simulation(synthetic_prices, sim_num)
            results.append(result)
        
        return results
    
    def _generate_optimized_synthetic_prices(self):
        """Generate synthetic price data with enhanced realism"""
        days = (self.end_date - self.start_date).days
        
        # Enhanced GBM with regime switching and volatility clustering
        dt = 1/365
        price_path = [self.btc_data['Close'].iloc[0]]
        
        # Regime parameters
        high_vol_regime = np.random.random() < 0.3  # 30% chance of high volatility period
        vol_multiplier = 1.5 if high_vol_regime else 1.0
        
        for day in range(1, days):
            # Volatility clustering
            if day > 10:
                recent_volatility = np.std([
                    np.log(price_path[i] / price_path[i-1]) 
                    for i in range(max(1, day-10), day)
                ])
                vol_adjustment = recent_volatility / self.return_std
            else:
                vol_adjustment = 1.0
            
            # Random walk with enhanced features
            random_shock = np.random.normal(0, 1)
            drift = self.return_mean + (0.001 if price_path[-1] < price_path[0] * 0.8 else 0)  # Mean reversion
            volatility = self.return_std * vol_multiplier * vol_adjustment
            
            # Price update
            price_change = drift * dt + volatility * np.sqrt(dt) * random_shock
            new_price = price_path[-1] * np.exp(price_change)
            price_path.append(new_price)
        
        return np.array(price_path)
    
    def _run_optimized_strategy_simulation(self, synthetic_prices, sim_num):
        """Run optimized strategy simulation with enhanced trade logic"""
        total_capital = 100000
        capital_deployed = 0
        total_btc = 0
        trades = 0
        last_trade_time = None
        trade_history = []
        
        # Calculate years for trade frequency
        years = (self.end_date - self.start_date).days / 365.25
        
        # Calculate indicators more efficiently
        price_series = pd.Series(synthetic_prices)
        rolling_highs = price_series.rolling(60, min_periods=1).max()
        drawdowns = (price_series - rolling_highs) / rolling_highs * 100
        
        # Enhanced Fear & Greed simulation
        fear_greed = np.clip(50 + drawdowns * 0.8 + np.random.normal(0, 5, len(drawdowns)), 0, 100)
        
        # Days since ATH calculation (vectorized)
        days_since_ath = np.zeros(len(synthetic_prices))
        for i in range(1, len(synthetic_prices)):
            if synthetic_prices[i] >= rolling_highs.iloc[i]:
                days_since_ath[i] = 0
            else:
                days_since_ath[i] = days_since_ath[i-1] + 1
        
        # Main trading loop
        for day, (price, drawdown, fg, days_ath) in enumerate(zip(
            synthetic_prices, drawdowns, fear_greed, days_since_ath
        )):
            # Enhanced cooldown check
            if last_trade_time is not None:
                cooldown_days = self.strategy_params['cooldown_hours'] / 24
                if day - last_trade_time < cooldown_days:
                    continue
            
            # OPTIMIZED entry criteria (more lenient)
            if (drawdown <= self.strategy_params['min_drawdown'] and
                fg <= self.strategy_params['max_fear_greed'] and
                days_ath >= self.strategy_params['min_days_since_ath']):
                
                # Calculate Fibonacci levels with optimized parameters
                lookback_start = max(0, day - 90)  # Increased lookback
                recent_high = price_series.iloc[lookback_start:day+1].max()
                recent_low = price_series.iloc[lookback_start:day+1].min()
                price_range = recent_high - recent_low
                
                # More lenient range requirement
                if price_range > recent_high * self.strategy_params['min_range_pct']:
                    
                    # Find best Fibonacci opportunity with enhanced scoring
                    best_score = 0
                    best_level = None
                    
                    for level_name, level_config in self.strategy_params['fibonacci_levels'].items():
                        fib_price = recent_high - (price_range * level_config['ratio'])
                        distance_pct = (price - fib_price) / fib_price * 100
                        
                        # Enhanced entry windows
                        entry_window = level_config['entry_window']
                        if entry_window[0] <= distance_pct <= entry_window[1]:
                            # Dynamic leverage calculation
                            base_lev = self.strategy_params['base_leverage']
                            drawdown_boost = abs(drawdown) * 0.25
                            fear_boost = (self.strategy_params['max_fear_greed'] - fg) * 0.15
                            time_boost = min(days_ath / 30, 1) * 0.5
                            
                            leverage = min(
                                base_lev + drawdown_boost + fear_boost + time_boost,
                                self.strategy_params['max_leverage']
                            )
                            
                            # Enhanced scoring with multiple factors
                            momentum_factor = 1 + (abs(drawdown) / 50)
                            volatility_factor = 1 + (price_range / recent_high)
                            time_factor = 1 + (min(days_ath, 60) / 60) * 0.3
                            
                            score = (level_config['base_multiplier'] * 
                                   leverage * 
                                   abs(distance_pct) * 
                                   momentum_factor *
                                   volatility_factor *
                                   time_factor)
                            
                            if score > best_score:
                                best_score = score
                                best_level = {
                                    'multiplier': level_config['base_multiplier'],
                                    'leverage': leverage,
                                    'level_name': level_name
                                }
                    
                    # Execute trade with optimized threshold
                    if best_level and best_score > self.strategy_params['score_threshold']:
                        remaining_capital = total_capital - capital_deployed
                        if remaining_capital > self.strategy_params['min_capital']:
                            
                            # Optimized position sizing
                            max_position = remaining_capital * self.strategy_params['max_position_pct']
                            base_position = min(max_position, remaining_capital * 0.12)
                            leverage = best_level['leverage']
                            total_position = base_position * leverage
                            
                            btc_acquired = total_position / price
                            
                            # Update portfolio
                            capital_deployed += base_position
                            total_btc += btc_acquired
                            trades += 1
                            last_trade_time = day
                            
                            # Track trade
                            trade_history.append({
                                'day': day,
                                'price': price,
                                'level': best_level['level_name'],
                                'leverage': leverage,
                                'position': base_position,
                                'btc': btc_acquired,
                                'drawdown': drawdown,
                                'score': best_score
                            })
        
        # Calculate final performance metrics
        if total_btc > 0:
            final_value = total_btc * synthetic_prices[-1]
            total_return = (final_value - capital_deployed) / capital_deployed
            annual_return = (final_value / capital_deployed) ** (1 / years) - 1
            
            # Buy & Hold comparison
            start_price = synthetic_prices[0]
            buy_hold_btc = total_capital / start_price
            buy_hold_final = buy_hold_btc * synthetic_prices[-1]
            buy_hold_return = (buy_hold_final - total_capital) / total_capital
            buy_hold_annual = (buy_hold_final / total_capital) ** (1 / years) - 1
            
            outperformance = annual_return - buy_hold_annual
            
            # Risk metrics
            if len(trade_history) > 1:
                trade_returns = [t['btc'] * synthetic_prices[-1] / t['position'] - 1 for t in trade_history]
                volatility = np.std(trade_returns) if trade_returns else 0
                sharpe = annual_return / volatility if volatility > 0 else 0
            else:
                volatility = sharpe = 0
                
        else:
            total_return = annual_return = outperformance = 0
            buy_hold_annual = final_value = volatility = sharpe = 0
        
        return {
            'simulation': sim_num,
            'trades': trades,
            'capital_deployed': capital_deployed,
            'total_btc': total_btc,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'buy_hold_annual': buy_hold_annual,
            'outperformance': outperformance,
            'max_drawdown': drawdowns.min(),
            'trades_per_year': trades / years,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'avg_leverage': np.mean([t['leverage'] for t in trade_history]) if trade_history else 0,
            'capital_efficiency': capital_deployed / total_capital if total_capital > 0 else 0
        }
    
    def _calculate_risk_metrics(self):
        """Calculate comprehensive risk metrics for optimized strategy"""
        annual_returns = self.results['annual_return']
        trades_per_year = self.results['trades_per_year']
        
        self.risk_metrics = {
            # Return Statistics
            'mean_annual_return': annual_returns.mean(),
            'median_annual_return': annual_returns.median(),
            'std_annual_return': annual_returns.std(),
            'min_annual_return': annual_returns.min(),
            'max_annual_return': annual_returns.max(),
            
            # Enhanced Risk Metrics
            'sharpe_ratio': annual_returns.mean() / annual_returns.std() if annual_returns.std() > 0 else 0,
            'sortino_ratio': annual_returns.mean() / annual_returns[annual_returns < 0].std() if len(annual_returns[annual_returns < 0]) > 0 else np.inf,
            'calmar_ratio': annual_returns.mean() / abs(self.results['max_drawdown'].mean()) if self.results['max_drawdown'].mean() != 0 else np.inf,
            
            # Value at Risk
            'var_95': np.percentile(annual_returns, 5),
            'var_99': np.percentile(annual_returns, 1),
            'cvar_95': annual_returns[annual_returns <= np.percentile(annual_returns, 5)].mean(),
            'cvar_99': annual_returns[annual_returns <= np.percentile(annual_returns, 1)].mean(),
            
            # Performance Probabilities
            'prob_positive': (annual_returns > 0).mean(),
            'prob_beat_target': (annual_returns > 2.454).mean(),  # +245.4% target
            'prob_beat_buy_hold': (self.results['outperformance'] > 0).mean(),
            'prob_beat_150': (annual_returns > 1.5).mean(),  # 150% return
            'prob_beat_200': (annual_returns > 2.0).mean(),  # 200% return
            
            # Trading Statistics (Enhanced)
            'mean_trades_per_year': trades_per_year.mean(),
            'median_trades_per_year': trades_per_year.median(),
            'optimal_frequency': ((trades_per_year >= 15) & (trades_per_year <= 20)).mean(),
            'high_frequency': (trades_per_year >= 20).mean(),
            'low_frequency': (trades_per_year < 10).mean(),
            
            # Capital Efficiency
            'mean_capital_efficiency': self.results['capital_efficiency'].mean(),
            'mean_leverage': self.results['avg_leverage'].mean(),
            
            # Percentiles
            'return_percentiles': {
                '10th': np.percentile(annual_returns, 10),
                '25th': np.percentile(annual_returns, 25),
                '75th': np.percentile(annual_returns, 75),
                '90th': np.percentile(annual_returns, 90)
            }
        }
    
    def _stress_test_scenarios(self):
        """Enhanced stress testing scenarios"""
        print(f"\n🔥 ENHANCED STRESS TEST SCENARIOS")
        
        stress_scenarios = {
            'crypto_winter_2022': {'return_shock': -0.15, 'volatility_multiplier': 2.5, 'duration_days': 180},
            'black_swan': {'return_shock': -0.30, 'volatility_multiplier': 4.0, 'duration_days': 30},
            'extended_bear': {'return_shock': -0.08, 'volatility_multiplier': 1.8, 'duration_days': 730},
            'flash_crash': {'return_shock': -0.25, 'volatility_multiplier': 3.5, 'duration_days': 7},
            'macro_crisis': {'return_shock': -0.12, 'volatility_multiplier': 2.2, 'duration_days': 365}
        }
        
        self.stress_results = {}
        
        for scenario_name, scenario in stress_scenarios.items():
            stress_results = []
            
            for _ in range(100):  # Run 100 stress tests per scenario
                # Generate stressed synthetic prices
                synthetic_prices = self._generate_optimized_synthetic_prices()
                
                # Apply stress scenario
                shock_start = len(synthetic_prices) // 3
                shock_end = min(shock_start + scenario['duration_days'], len(synthetic_prices))
                
                for day in range(shock_start, shock_end):
                    daily_shock = scenario['return_shock'] / scenario['duration_days']
                    volatility_boost = (scenario['volatility_multiplier'] - 1) * self.return_std
                    
                    shock = daily_shock + np.random.normal(0, volatility_boost)
                    synthetic_prices[day:] *= (1 + shock)
                
                # Run strategy simulation
                result = self._run_optimized_strategy_simulation(synthetic_prices, 0)
                stress_results.append(result['annual_return'])
            
            # Calculate stress metrics
            stress_returns = np.array(stress_results)
            self.stress_results[scenario_name] = {
                'mean_return': stress_returns.mean(),
                'worst_case': stress_returns.min(),
                'var_95': np.percentile(stress_returns, 5),
                'prob_positive': (stress_returns > 0).mean()
            }
            
            print(f"\n   {scenario_name.replace('_', ' ').title()}:")
            print(f"      Mean Return: {self.stress_results[scenario_name]['mean_return']:+.1%}")
            print(f"      Worst Case: {self.stress_results[scenario_name]['worst_case']:+.1%}")
            print(f"      VaR 95%: {self.stress_results[scenario_name]['var_95']:+.1%}")
            print(f"      Prob Positive: {self.stress_results[scenario_name]['prob_positive']:.1%}")
    
    def _display_results(self):
        """Display comprehensive optimized results"""
        print(f"\n🎲 OPTIMIZED MONTE CARLO ANALYSIS RESULTS")
        print("=" * 70)
        
        print(f"\n📊 RETURN DISTRIBUTION:")
        print(f"   Mean Annual Return: {self.risk_metrics['mean_annual_return']:+.1%}")
        print(f"   Median Annual Return: {self.risk_metrics['median_annual_return']:+.1%}")
        print(f"   Standard Deviation: {self.risk_metrics['std_annual_return']:.1%}")
        print(f"   Best Case: {self.risk_metrics['max_annual_return']:+.1%}")
        print(f"   Worst Case: {self.risk_metrics['min_annual_return']:+.1%}")
        
        print(f"\n📊 ENHANCED RISK METRICS:")
        print(f"   Sharpe Ratio: {self.risk_metrics['sharpe_ratio']:.2f}")
        print(f"   Sortino Ratio: {self.risk_metrics['sortino_ratio']:.2f}")
        print(f"   Calmar Ratio: {self.risk_metrics['calmar_ratio']:.2f}")
        print(f"   Value at Risk (95%): {self.risk_metrics['var_95']:+.1%}")
        print(f"   Value at Risk (99%): {self.risk_metrics['var_99']:+.1%}")
        print(f"   Conditional VaR (95%): {self.risk_metrics['cvar_95']:+.1%}")
        print(f"   Conditional VaR (99%): {self.risk_metrics['cvar_99']:+.1%}")
        
        print(f"\n📊 SUCCESS PROBABILITIES:")
        print(f"   Probability of Positive Returns: {self.risk_metrics['prob_positive']:.1%}")
        print(f"   Probability of +150% Returns: {self.risk_metrics['prob_beat_150']:.1%}")
        print(f"   Probability of +200% Returns: {self.risk_metrics['prob_beat_200']:.1%}")
        print(f"   Probability of Beating Target (+245.4%): {self.risk_metrics['prob_beat_target']:.1%}")
        print(f"   Probability of Beating Buy & Hold: {self.risk_metrics['prob_beat_buy_hold']:.1%}")
        
        print(f"\n📊 OPTIMIZED TRADING CHARACTERISTICS:")
        print(f"   Mean Trades per Year: {self.risk_metrics['mean_trades_per_year']:.1f}")
        print(f"   Median Trades per Year: {self.risk_metrics['median_trades_per_year']:.1f}")
        print(f"   Optimal Frequency (15-20/year): {self.risk_metrics['optimal_frequency']:.1%}")
        print(f"   High Frequency (>20/year): {self.risk_metrics['high_frequency']:.1%}")
        print(f"   Mean Capital Efficiency: {self.risk_metrics['mean_capital_efficiency']:.1%}")
        print(f"   Average Leverage: {self.risk_metrics['mean_leverage']:.1f}x")
        
        # Risk grade calculation
        score = 0
        if self.risk_metrics['sharpe_ratio'] > 2.0: score += 2
        elif self.risk_metrics['sharpe_ratio'] > 1.5: score += 1
        
        if self.risk_metrics['prob_positive'] > 0.95: score += 2
        elif self.risk_metrics['prob_positive'] > 0.85: score += 1
        
        if self.risk_metrics['prob_beat_target'] > 0.20: score += 2
        elif self.risk_metrics['prob_beat_target'] > 0.10: score += 1
        
        if self.risk_metrics['optimal_frequency'] > 0.30: score += 2
        elif self.risk_metrics['optimal_frequency'] > 0.15: score += 1
        
        grade = ['D', 'C', 'B', 'A', 'A+'][min(score, 4)]
        
        print(f"\n🏆 OPTIMIZED RISK GRADE: {grade} 🎯")
        print(f"💡 ASSESSMENT: {'EXCELLENT' if score >= 6 else 'STRONG' if score >= 4 else 'MODERATE'} RISK PROFILE")
    
    def _create_optimized_visualizations(self):
        """Create enhanced visualizations for optimized analysis"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Optimized Monte Carlo Risk Analysis - Enhanced Nanpin Strategy', 
                        fontsize=16, fontweight='bold')
            
            # 1. Return distribution (enhanced)
            axes[0, 0].hist(self.results['annual_return'], bins=60, alpha=0.7, 
                           color='steelblue', edgecolor='black', density=True)
            axes[0, 0].axvline(self.risk_metrics['mean_annual_return'], color='red', 
                              linestyle='--', linewidth=2,
                              label=f'Mean: {self.risk_metrics["mean_annual_return"]:.1%}')
            axes[0, 0].axvline(2.454, color='green', linestyle='--', linewidth=2,
                              label='Target: 245.4%')
            axes[0, 0].axvline(self.risk_metrics['median_annual_return'], color='orange', 
                              linestyle=':', linewidth=2,
                              label=f'Median: {self.risk_metrics["median_annual_return"]:.1%}')
            axes[0, 0].set_title('Annual Return Distribution')
            axes[0, 0].set_xlabel('Annual Return')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Trade frequency vs Returns (optimized)
            valid_mask = (self.results['trades_per_year'] >= 0) & (self.results['trades_per_year'] <= 50)
            valid_trades = self.results['trades_per_year'][valid_mask]
            valid_returns = self.results['annual_return'][valid_mask]
            
            scatter = axes[0, 1].scatter(valid_trades, valid_returns, 
                                       c=self.results['capital_efficiency'][valid_mask],
                                       cmap='viridis', alpha=0.6, s=20)
            axes[0, 1].axvspan(15, 20, alpha=0.2, color='green', label='Optimal Frequency')
            axes[0, 1].axhline(2.454, color='green', linestyle='--', label='Target Return')
            axes[0, 1].axvline(valid_trades.mean(), color='red', linestyle='--', alpha=0.7,
                              label=f'Mean: {valid_trades.mean():.1f}/year')
            
            axes[0, 1].set_title('Trade Frequency vs Returns (Color: Capital Efficiency)')
            axes[0, 1].set_xlabel('Trades per Year')
            axes[0, 1].set_ylabel('Annual Return')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[0, 1], label='Capital Efficiency')
            
            # 3. Risk-Return scatter
            axes[0, 2].scatter(self.results['volatility'], self.results['annual_return'],
                              alpha=0.6, s=20, color='purple')
            axes[0, 2].set_title('Risk vs Return')
            axes[0, 2].set_xlabel('Volatility')
            axes[0, 2].set_ylabel('Annual Return')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. VaR comparison
            var_data = [
                self.risk_metrics['var_95'],
                self.risk_metrics['var_99'],
                self.risk_metrics['cvar_95'],
                self.risk_metrics['cvar_99']
            ]
            var_labels = ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%']
            colors = ['orange', 'red', 'darkorange', 'darkred']
            
            bars = axes[1, 0].bar(var_labels, var_data, color=colors, alpha=0.7)
            axes[1, 0].set_title('Value at Risk Metrics')
            axes[1, 0].set_ylabel('Annual Return')
            axes[1, 0].grid(True, alpha=0.3)
            
            for bar, value in zip(bars, var_data):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.1%}', ha='center', va='bottom')
            
            # 5. Performance percentiles
            percentiles = [10, 25, 50, 75, 90]
            percentile_values = [np.percentile(self.results['annual_return'], p) for p in percentiles]
            
            axes[1, 1].plot(percentiles, percentile_values, 'o-', linewidth=2, markersize=8)
            axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.7, label='Break-even')
            axes[1, 1].axhline(2.454, color='green', linestyle='--', alpha=0.7, label='Target')
            axes[1, 1].set_title('Return Percentiles')
            axes[1, 1].set_xlabel('Percentile')
            axes[1, 1].set_ylabel('Annual Return')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Leverage vs Returns
            axes[1, 2].scatter(self.results['avg_leverage'], self.results['annual_return'],
                              alpha=0.6, s=20, color='brown')
            axes[1, 2].set_title('Average Leverage vs Returns')
            axes[1, 2].set_xlabel('Average Leverage')
            axes[1, 2].set_ylabel('Annual Return')
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('monte_carlo_risk_analysis_optimized.png', dpi=300, bbox_inches='tight')
            print(f"\n📊 Enhanced visualizations saved to: monte_carlo_risk_analysis_optimized.png")
            
        except Exception as e:
            print(f"⚠️ Could not create visualizations: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Main execution function"""
    analyzer = OptimizedMonteCarloRiskAnalyzer()
    
    try:
        results, risk_metrics = await analyzer.run_monte_carlo_analysis()
        
        print(f"\n🎉 OPTIMIZED MONTE CARLO ANALYSIS COMPLETE!")
        print(f"✅ Generated comprehensive enhanced risk assessment")
        print(f"📊 Key insight: {risk_metrics['prob_beat_target']:.1%} probability of hitting +245.4% target")
        print(f"🎯 Trade frequency: {risk_metrics['mean_trades_per_year']:.1f} trades/year")
        print(f"💪 Optimal frequency achieved: {risk_metrics['optimal_frequency']:.1%} of simulations")
        
        return results, risk_metrics
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    asyncio.run(main())