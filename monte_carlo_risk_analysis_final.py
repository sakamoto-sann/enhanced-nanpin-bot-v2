#!/usr/bin/env python3
"""
🎲 Final Optimized Monte Carlo Risk Analysis for Nanpin Strategy
Fixed parallel processing and optimized for 15-20 trades/year
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FinalOptimizedMonteCarloRiskAnalyzer:
    def __init__(self, start_date="2020-01-01", end_date="2024-12-31"):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.simulations = 1000  # Focused on quality over quantity
        
        # OPTIMIZED Strategy parameters for 15-20 trades/year target
        self.strategy_params = {
            'min_drawdown': -12,  # More lenient for more trades
            'max_fear_greed': 45,  # Higher threshold
            'min_days_since_ath': 3,  # Reduced for more opportunities
            'fibonacci_levels': {
                '23.6%': {'ratio': 0.236, 'base_multiplier': 2.0, 'entry_window': (-8.0, -0.5)},
                '38.2%': {'ratio': 0.382, 'base_multiplier': 3.0, 'entry_window': (-12.0, -0.5)},
                '50.0%': {'ratio': 0.500, 'base_multiplier': 4.5, 'entry_window': (-15.0, -0.5)},
                '61.8%': {'ratio': 0.618, 'base_multiplier': 7.5, 'entry_window': (-20.0, -0.5)},
                '78.6%': {'ratio': 0.786, 'base_multiplier': 12.0, 'entry_window': (-25.0, -0.5)}
            },
            'base_leverage': 3.0,  # Slightly higher
            'max_leverage': 16.0,
            'cooldown_hours': 24,  # Reduced for more frequent trades
            'min_range_pct': 0.035,  # Slightly more lenient
            'score_threshold': 3.5,  # Lower threshold for more trades
            'max_position_pct': 0.20,  # Slightly higher
            'min_capital': 50  # Lower minimum
        }
        
        self.results = []
        self.risk_metrics = {}
        
        print(f"🚀 Final Optimized Monte Carlo Analyzer initialized")
        print(f"   Simulations: {self.simulations}")
        print(f"   Target: 15-20 trades/year with enhanced returns")
    
    async def run_monte_carlo_analysis(self):
        """Run final optimized Monte Carlo analysis"""
        print(f"🎲 FINAL OPTIMIZED MONTE CARLO RISK ANALYSIS")
        print("=" * 70)
        print(f"Simulations: {self.simulations}")
        print(f"Strategy: Final Enhanced Goldilocks Plus Nanpin")
        print(f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        
        # Load historical data
        await self._load_data()
        
        # Run simulations (sequential for reliability)
        await self._run_simulations()
        
        # Calculate risk metrics
        self._calculate_risk_metrics()
        
        # Run stress tests
        self._stress_test_scenarios()
        
        # Display results
        self._display_results()
        
        # Create visualizations
        self._create_final_visualizations()
        
        return self.results, self.risk_metrics
    
    async def _load_data(self):
        """Load and preprocess Bitcoin data"""
        try:
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
    
    async def _run_simulations(self):
        """Run simulations sequentially for reliability"""
        print(f"\n🔄 Running {self.simulations} optimized Monte Carlo simulations...")
        
        results = []
        for sim_num in range(self.simulations):
            if sim_num % 100 == 0:
                print(f"   Simulation {sim_num + 1}/{self.simulations}")
            
            # Generate synthetic prices
            synthetic_prices = self._generate_enhanced_synthetic_prices()
            
            # Run strategy simulation
            result = self._run_final_strategy_simulation(synthetic_prices, sim_num)
            results.append(result)
        
        self.results = pd.DataFrame(results)
        print(f"✅ Completed {len(self.results)} simulations")
        print(f"   Average trades per year: {self.results['trades_per_year'].mean():.1f}")
        print(f"   Average annual return: {self.results['annual_return'].mean():.1%}")
    
    def _generate_enhanced_synthetic_prices(self):
        """Generate realistic synthetic price data"""
        days = (self.end_date - self.start_date).days
        
        # Enhanced GBM with volatility clustering and mean reversion
        dt = 1/365
        price_path = [self.btc_data['Close'].iloc[0]]
        
        # Market regime simulation
        bull_regime = np.random.random() < 0.6  # 60% chance of bull market
        bear_phase_length = np.random.randint(90, 270) if not bull_regime else 0
        
        for day in range(1, days):
            # Dynamic volatility with clustering
            if day > 20:
                recent_returns = [np.log(price_path[i] / price_path[i-1]) for i in range(max(1, day-20), day)]
                recent_vol = np.std(recent_returns)
                vol_multiplier = 0.7 + 1.3 * (recent_vol / self.return_std)
            else:
                vol_multiplier = 1.0
            
            # Market regime effects
            if not bull_regime and day < bear_phase_length:
                drift_adjustment = -0.002  # Bear market bias
                vol_multiplier *= 1.3
            else:
                drift_adjustment = 0.001  # Slight bull bias
            
            # Mean reversion component
            current_level = price_path[-1] / price_path[0]
            if current_level > 2.0:  # Price doubled
                mean_reversion = -0.001
            elif current_level < 0.7:  # Price dropped 30%
                mean_reversion = 0.002
            else:
                mean_reversion = 0
            
            # Generate price change
            drift = self.return_mean + drift_adjustment + mean_reversion
            volatility = self.return_std * vol_multiplier
            random_shock = np.random.normal(0, 1)
            
            price_change = drift * dt + volatility * np.sqrt(dt) * random_shock
            new_price = price_path[-1] * np.exp(price_change)
            price_path.append(max(new_price, price_path[0] * 0.1))  # Floor at 10% of starting price
        
        return np.array(price_path)
    
    def _run_final_strategy_simulation(self, synthetic_prices, sim_num):
        """Run final optimized strategy simulation"""
        total_capital = 100000
        capital_deployed = 0
        total_btc = 0
        trades = 0
        last_trade_time = None
        trade_history = []
        
        # Calculate years for trade frequency
        years = (self.end_date - self.start_date).days / 365.25
        
        # Calculate indicators efficiently
        price_series = pd.Series(synthetic_prices)
        rolling_highs = price_series.rolling(60, min_periods=1).max()
        drawdowns = (price_series - rolling_highs) / rolling_highs * 100
        
        # Enhanced Fear & Greed simulation with more realistic distribution
        base_fg = 50 + drawdowns * 0.7
        noise = np.random.normal(0, 8, len(drawdowns))  # More noise for realism
        fear_greed = np.clip(base_fg + noise, 5, 95)  # Wider range
        
        # Days since ATH calculation
        days_since_ath = np.zeros(len(synthetic_prices))
        for i in range(1, len(synthetic_prices)):
            if synthetic_prices[i] >= rolling_highs.iloc[i]:
                days_since_ath[i] = 0
            else:
                days_since_ath[i] = days_since_ath[i-1] + 1
        
        # Main trading loop with enhanced logic
        for day, (price, drawdown, fg, days_ath) in enumerate(zip(
            synthetic_prices, drawdowns, fear_greed, days_since_ath
        )):
            
            # Enhanced cooldown check
            if last_trade_time is not None:
                cooldown_days = self.strategy_params['cooldown_hours'] / 24
                if day - last_trade_time < cooldown_days:
                    continue
            
            # OPTIMIZED entry criteria
            if (drawdown <= self.strategy_params['min_drawdown'] and
                fg <= self.strategy_params['max_fear_greed'] and
                days_ath >= self.strategy_params['min_days_since_ath']):
                
                # Enhanced Fibonacci analysis
                lookback_start = max(0, day - 80)  # Balanced lookback
                recent_high = price_series.iloc[lookback_start:day+1].max()
                recent_low = price_series.iloc[lookback_start:day+1].min()
                price_range = recent_high - recent_low
                
                if price_range > recent_high * self.strategy_params['min_range_pct']:
                    
                    # Find best Fibonacci opportunity
                    best_score = 0
                    best_level = None
                    
                    for level_name, level_config in self.strategy_params['fibonacci_levels'].items():
                        fib_price = recent_high - (price_range * level_config['ratio'])
                        distance_pct = (price - fib_price) / fib_price * 100
                        
                        # Check entry window
                        entry_window = level_config['entry_window']
                        if entry_window[0] <= distance_pct <= entry_window[1]:
                            
                            # Dynamic leverage calculation
                            base_lev = self.strategy_params['base_leverage']
                            drawdown_boost = abs(drawdown) * 0.2
                            fear_boost = (self.strategy_params['max_fear_greed'] - fg) * 0.15
                            time_boost = min(days_ath / 40, 1) * 0.4
                            
                            leverage = min(
                                base_lev + drawdown_boost + fear_boost + time_boost,
                                self.strategy_params['max_leverage']
                            )
                            
                            # Enhanced scoring
                            momentum_factor = 1 + (abs(drawdown) / 30)
                            volatility_factor = 1 + (price_range / recent_high) * 0.5
                            time_factor = 1 + (min(days_ath, 50) / 50) * 0.3
                            fear_factor = 1 + (self.strategy_params['max_fear_greed'] - fg) / 100
                            
                            score = (level_config['base_multiplier'] * 
                                   leverage * 
                                   abs(distance_pct) * 
                                   momentum_factor *
                                   volatility_factor *
                                   time_factor *
                                   fear_factor)
                            
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
                            
                            # Enhanced position sizing
                            base_position_pct = min(0.15 + abs(drawdown) * 0.005, self.strategy_params['max_position_pct'])
                            base_position = min(remaining_capital * base_position_pct, remaining_capital * 0.25)
                            leverage = best_level['leverage']
                            total_position = base_position * leverage
                            
                            btc_acquired = total_position / price
                            
                            # Update portfolio
                            capital_deployed += base_position
                            total_btc += btc_acquired
                            trades += 1
                            last_trade_time = day
                            
                            # Track trade details
                            trade_history.append({
                                'day': day,
                                'price': price,
                                'level': best_level['level_name'],
                                'leverage': leverage,
                                'position': base_position,
                                'btc': btc_acquired,
                                'drawdown': drawdown,
                                'fear_greed': fg,
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
            buy_hold_annual = (buy_hold_final / total_capital) ** (1 / years) - 1
            
            outperformance = annual_return - buy_hold_annual
            
            # Enhanced metrics
            if len(trade_history) > 0:
                avg_leverage = np.mean([t['leverage'] for t in trade_history])
                capital_efficiency = capital_deployed / total_capital
                avg_drawdown_at_entry = np.mean([abs(t['drawdown']) for t in trade_history])
            else:
                avg_leverage = capital_efficiency = avg_drawdown_at_entry = 0
                
        else:
            (final_value, total_return, annual_return, buy_hold_annual, outperformance, 
             avg_leverage, capital_efficiency, avg_drawdown_at_entry) = [0] * 8
        
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
            'avg_leverage': avg_leverage,
            'capital_efficiency': capital_efficiency,
            'avg_drawdown_at_entry': avg_drawdown_at_entry
        }
    
    def _calculate_risk_metrics(self):
        """Calculate comprehensive risk metrics"""
        annual_returns = self.results['annual_return']
        trades_per_year = self.results['trades_per_year']
        
        self.risk_metrics = {
            # Return Statistics
            'mean_annual_return': annual_returns.mean(),
            'median_annual_return': annual_returns.median(),
            'std_annual_return': annual_returns.std(),
            'min_annual_return': annual_returns.min(),
            'max_annual_return': annual_returns.max(),
            
            # Risk Metrics
            'sharpe_ratio': annual_returns.mean() / annual_returns.std() if annual_returns.std() > 0 else 0,
            'sortino_ratio': annual_returns.mean() / annual_returns[annual_returns < 0].std() if len(annual_returns[annual_returns < 0]) > 0 else np.inf,
            
            # Value at Risk
            'var_95': np.percentile(annual_returns, 5),
            'var_99': np.percentile(annual_returns, 1),
            'cvar_95': annual_returns[annual_returns <= np.percentile(annual_returns, 5)].mean(),
            'cvar_99': annual_returns[annual_returns <= np.percentile(annual_returns, 1)].mean(),
            
            # Performance Probabilities
            'prob_positive': (annual_returns > 0).mean(),
            'prob_beat_target': (annual_returns > 2.454).mean(),  # +245.4% target
            'prob_beat_buy_hold': (self.results['outperformance'] > 0).mean(),
            'prob_beat_150': (annual_returns > 1.5).mean(),
            'prob_beat_200': (annual_returns > 2.0).mean(),
            'prob_beat_300': (annual_returns > 3.0).mean(),
            
            # Trading Statistics
            'mean_trades_per_year': trades_per_year.mean(),
            'median_trades_per_year': trades_per_year.median(),
            'optimal_frequency': ((trades_per_year >= 15) & (trades_per_year <= 20)).mean(),
            'high_frequency': (trades_per_year >= 20).mean(),
            'target_frequency': ((trades_per_year >= 12) & (trades_per_year <= 25)).mean(),
            
            # Strategy Efficiency
            'mean_capital_efficiency': self.results['capital_efficiency'].mean(),
            'mean_leverage': self.results['avg_leverage'].mean(),
            'mean_entry_drawdown': self.results['avg_drawdown_at_entry'].mean(),
            
            # Percentiles
            'return_percentiles': {
                '10th': np.percentile(annual_returns, 10),
                '25th': np.percentile(annual_returns, 25),
                '75th': np.percentile(annual_returns, 75),
                '90th': np.percentile(annual_returns, 90)
            }
        }
    
    def _stress_test_scenarios(self):
        """Run stress test scenarios"""
        print(f"\n🔥 STRESS TEST SCENARIOS")
        
        scenarios = {
            'crypto_winter': {'return_shock': -0.12, 'volatility_multiplier': 2.0},
            'black_swan': {'return_shock': -0.25, 'volatility_multiplier': 3.0},
            'extended_bear': {'return_shock': -0.08, 'volatility_multiplier': 1.5},
            'flash_crash': {'return_shock': -0.20, 'volatility_multiplier': 2.5}
        }
        
        self.stress_results = {}
        
        for scenario_name, scenario in scenarios.items():
            stress_results = []
            
            for _ in range(50):  # 50 stress tests per scenario
                synthetic_prices = self._generate_enhanced_synthetic_prices()
                
                # Apply stress
                shock_start = len(synthetic_prices) // 3
                shock_duration = 120  # 4 months
                shock_end = min(shock_start + shock_duration, len(synthetic_prices))
                
                for day in range(shock_start, shock_end):
                    daily_shock = scenario['return_shock'] / shock_duration
                    vol_shock = (scenario['volatility_multiplier'] - 1) * self.return_std * np.random.normal(0, 1)
                    total_shock = daily_shock + vol_shock
                    synthetic_prices[day:] *= (1 + total_shock)
                
                result = self._run_final_strategy_simulation(synthetic_prices, 0)
                stress_results.append(result['annual_return'])
            
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
        """Display comprehensive results"""
        print(f"\n🎲 FINAL OPTIMIZED MONTE CARLO ANALYSIS RESULTS")
        print("=" * 70)
        
        print(f"\n📊 RETURN DISTRIBUTION:")
        print(f"   Mean Annual Return: {self.risk_metrics['mean_annual_return']:+.1%}")
        print(f"   Median Annual Return: {self.risk_metrics['median_annual_return']:+.1%}")
        print(f"   Standard Deviation: {self.risk_metrics['std_annual_return']:.1%}")
        print(f"   Best Case: {self.risk_metrics['max_annual_return']:+.1%}")
        print(f"   Worst Case: {self.risk_metrics['min_annual_return']:+.1%}")
        
        print(f"\n📊 RISK METRICS:")
        print(f"   Sharpe Ratio: {self.risk_metrics['sharpe_ratio']:.2f}")
        print(f"   Sortino Ratio: {self.risk_metrics['sortino_ratio']:.2f}")
        print(f"   Value at Risk (95%): {self.risk_metrics['var_95']:+.1%}")
        print(f"   Value at Risk (99%): {self.risk_metrics['var_99']:+.1%}")
        print(f"   Conditional VaR (95%): {self.risk_metrics['cvar_95']:+.1%}")
        
        print(f"\n📊 SUCCESS PROBABILITIES:")
        print(f"   Probability of Positive Returns: {self.risk_metrics['prob_positive']:.1%}")
        print(f"   Probability of +150% Returns: {self.risk_metrics['prob_beat_150']:.1%}")
        print(f"   Probability of +200% Returns: {self.risk_metrics['prob_beat_200']:.1%}")
        print(f"   Probability of +300% Returns: {self.risk_metrics['prob_beat_300']:.1%}")
        print(f"   Probability of Beating Target (+245.4%): {self.risk_metrics['prob_beat_target']:.1%}")
        print(f"   Probability of Beating Buy & Hold: {self.risk_metrics['prob_beat_buy_hold']:.1%}")
        
        print(f"\n📊 OPTIMIZED TRADING CHARACTERISTICS:")
        print(f"   Mean Trades per Year: {self.risk_metrics['mean_trades_per_year']:.1f}")
        print(f"   Median Trades per Year: {self.risk_metrics['median_trades_per_year']:.1f}")
        print(f"   Target Frequency (12-25/year): {self.risk_metrics['target_frequency']:.1%}")
        print(f"   Optimal Frequency (15-20/year): {self.risk_metrics['optimal_frequency']:.1%}")
        print(f"   High Frequency (>20/year): {self.risk_metrics['high_frequency']:.1%}")
        print(f"   Mean Capital Efficiency: {self.risk_metrics['mean_capital_efficiency']:.1%}")
        print(f"   Average Leverage: {self.risk_metrics['mean_leverage']:.1f}x")
        print(f"   Average Entry Drawdown: {self.risk_metrics['mean_entry_drawdown']:.1f}%")
        
        # Enhanced risk grade
        score = 0
        if self.risk_metrics['sharpe_ratio'] > 2.0: score += 2
        elif self.risk_metrics['sharpe_ratio'] > 1.5: score += 1
        
        if self.risk_metrics['prob_positive'] > 0.95: score += 2
        elif self.risk_metrics['prob_positive'] > 0.85: score += 1
        
        if self.risk_metrics['prob_beat_target'] > 0.25: score += 2
        elif self.risk_metrics['prob_beat_target'] > 0.15: score += 1
        
        if self.risk_metrics['target_frequency'] > 0.50: score += 2
        elif self.risk_metrics['target_frequency'] > 0.30: score += 1
        
        if self.risk_metrics['mean_annual_return'] > 2.0: score += 1
        
        grade = ['D', 'C', 'B', 'A', 'A+'][min(score, 4)]
        assessment = 'EXCELLENT' if score >= 7 else 'STRONG' if score >= 5 else 'GOOD' if score >= 3 else 'MODERATE'
        
        print(f"\n🏆 FINAL OPTIMIZATION GRADE: {grade} 🎯")
        print(f"💡 ASSESSMENT: {assessment} OPTIMIZED STRATEGY")
    
    def _create_final_visualizations(self):
        """Create final optimized visualizations"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Final Optimized Monte Carlo Analysis - Enhanced Nanpin Strategy', 
                        fontsize=16, fontweight='bold')
            
            # 1. Return distribution
            axes[0, 0].hist(self.results['annual_return'], bins=50, alpha=0.7, 
                           color='steelblue', edgecolor='black')
            axes[0, 0].axvline(self.risk_metrics['mean_annual_return'], color='red', 
                              linestyle='--', linewidth=2,
                              label=f'Mean: {self.risk_metrics["mean_annual_return"]:.1%}')
            axes[0, 0].axvline(2.454, color='green', linestyle='--', linewidth=2,
                              label='Target: 245.4%')
            axes[0, 0].set_title('Annual Return Distribution')
            axes[0, 0].set_xlabel('Annual Return')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Trade frequency optimization
            valid_mask = self.results['trades_per_year'] > 0
            valid_trades = self.results['trades_per_year'][valid_mask]
            valid_returns = self.results['annual_return'][valid_mask]
            
            axes[0, 1].scatter(valid_trades, valid_returns, alpha=0.6, s=30, color='purple')
            axes[0, 1].axvspan(15, 20, alpha=0.2, color='green', label='Optimal (15-20)')
            axes[0, 1].axvspan(12, 25, alpha=0.1, color='blue', label='Target (12-25)')
            axes[0, 1].axhline(2.454, color='green', linestyle='--', label='Target Return')
            axes[0, 1].axvline(valid_trades.mean(), color='red', linestyle='--',
                              label=f'Mean: {valid_trades.mean():.1f}/year')
            axes[0, 1].set_title('Optimized Trade Frequency vs Returns')
            axes[0, 1].set_xlabel('Trades per Year')
            axes[0, 1].set_ylabel('Annual Return')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. VaR metrics
            var_data = [
                self.risk_metrics['var_95'],
                self.risk_metrics['var_99'],
                self.risk_metrics['cvar_95'],
                self.risk_metrics['cvar_99']
            ]
            var_labels = ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%']
            colors = ['orange', 'red', 'darkorange', 'darkred']
            
            bars = axes[0, 2].bar(var_labels, var_data, color=colors, alpha=0.7)
            axes[0, 2].set_title('Risk Metrics')
            axes[0, 2].set_ylabel('Annual Return')
            axes[0, 2].grid(True, alpha=0.3)
            
            for bar, value in zip(bars, var_data):
                height = bar.get_height()
                axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.1%}', ha='center', va='bottom')
            
            # 4. Capital efficiency
            axes[1, 0].scatter(self.results['capital_efficiency'], self.results['annual_return'],
                              alpha=0.6, s=30, color='brown')
            axes[1, 0].set_title('Capital Efficiency vs Returns')
            axes[1, 0].set_xlabel('Capital Efficiency')
            axes[1, 0].set_ylabel('Annual Return')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Leverage analysis
            valid_leverage = self.results['avg_leverage'][self.results['avg_leverage'] > 0]
            valid_returns_lev = self.results['annual_return'][self.results['avg_leverage'] > 0]
            
            axes[1, 1].scatter(valid_leverage, valid_returns_lev, alpha=0.6, s=30, color='darkgreen')
            axes[1, 1].set_title('Average Leverage vs Returns')
            axes[1, 1].set_xlabel('Average Leverage')
            axes[1, 1].set_ylabel('Annual Return')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Performance percentiles
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            percentile_values = [np.percentile(self.results['annual_return'], p) for p in percentiles]
            
            axes[1, 2].plot(percentiles, percentile_values, 'o-', linewidth=2, markersize=6, color='navy')
            axes[1, 2].axhline(0, color='red', linestyle='--', alpha=0.7, label='Break-even')
            axes[1, 2].axhline(2.454, color='green', linestyle='--', alpha=0.7, label='Target')
            axes[1, 2].set_title('Return Percentiles')
            axes[1, 2].set_xlabel('Percentile')
            axes[1, 2].set_ylabel('Annual Return')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('monte_carlo_risk_analysis_final.png', dpi=300, bbox_inches='tight')
            print(f"\n📊 Final optimized visualizations saved to: monte_carlo_risk_analysis_final.png")
            
        except Exception as e:
            print(f"⚠️ Could not create visualizations: {e}")

async def main():
    """Main execution function"""
    analyzer = FinalOptimizedMonteCarloRiskAnalyzer()
    
    try:
        results, risk_metrics = await analyzer.run_monte_carlo_analysis()
        
        print(f"\n🎉 FINAL OPTIMIZED MONTE CARLO ANALYSIS COMPLETE!")
        print(f"✅ Generated enhanced risk assessment with optimized trade frequency")
        print(f"📊 Key insights:")
        print(f"   • Target achievement: {risk_metrics['prob_beat_target']:.1%} probability of +245.4%")
        print(f"   • Trade frequency: {risk_metrics['mean_trades_per_year']:.1f} trades/year")
        print(f"   • Optimal frequency: {risk_metrics['optimal_frequency']:.1%} of simulations")
        print(f"   • Target frequency: {risk_metrics['target_frequency']:.1%} of simulations")
        print(f"   • Mean return: {risk_metrics['mean_annual_return']:.1%}")
        
        return results, risk_metrics
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    asyncio.run(main())