#!/usr/bin/env python3
"""
🎲 Monte Carlo Risk Analysis for Nanpin Strategies
Advanced risk simulation and stress testing
"""

import asyncio
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MonteCarloRiskAnalyzer:
    """Monte Carlo simulations for Nanpin strategy risk analysis"""
    
    def __init__(self, start_date="2020-01-01", end_date="2024-12-31"):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.simulations = 1000  # Number of Monte Carlo runs
        
        # Strategy parameters (Goldilocks Plus - best performing)
        self.strategy_params = {
            'min_drawdown': -18,
            'max_fear_greed': 35,
            'min_days_since_ath': 7,
            'fibonacci_levels': {
                '23.6%': {'ratio': 0.236, 'base_multiplier': 2},
                '38.2%': {'ratio': 0.382, 'base_multiplier': 3},
                '50.0%': {'ratio': 0.500, 'base_multiplier': 5},
                '61.8%': {'ratio': 0.618, 'base_multiplier': 8},
                '78.6%': {'ratio': 0.786, 'base_multiplier': 13}
            },
            'base_leverage': 3.0,
            'max_leverage': 18.0,
            'cooldown_hours': 48
        }
        
        self.results = []
        self.risk_metrics = {}
        
    async def run_monte_carlo_analysis(self):
        """Run comprehensive Monte Carlo risk analysis"""
        try:
            print("🎲 MONTE CARLO RISK ANALYSIS")
            print("=" * 65)
            print(f"Simulations: {self.simulations}")
            print(f"Strategy: Goldilocks Plus Nanpin")
            print(f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
            
            # Load historical data
            await self._load_data()
            
            # Run simulations
            print(f"\n🔄 Running {self.simulations} Monte Carlo simulations...")
            await self._run_simulations()
            
            # Calculate risk metrics
            self._calculate_risk_metrics()
            
            # Generate stress test scenarios
            self._stress_test_scenarios()
            
            # Display results
            self._display_results()
            
            # Generate visualizations
            self._create_visualizations()
            
            return self.risk_metrics
            
        except Exception as e:
            print(f"❌ Monte Carlo analysis failed: {e}")
            return {}
    
    async def _load_data(self):
        """Load and prepare historical data"""
        btc = yf.Ticker("BTC-USD")
        data = btc.history(
            start=(self.start_date - timedelta(days=200)).strftime("%Y-%m-%d"),
            end=self.end_date.strftime("%Y-%m-%d"),
            interval="1d"
        )
        
        self.btc_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        self.btc_data.index = pd.to_datetime(self.btc_data.index.date)
        self.btc_data = self.btc_data.dropna()
        
        # Calculate returns for simulation
        self.daily_returns = self.btc_data['Close'].pct_change().dropna()
        self.return_mean = self.daily_returns.mean()
        self.return_std = self.daily_returns.std()
        
        print(f"✅ Data loaded: {len(self.btc_data)} days")
        print(f"   Historical volatility: {self.return_std * np.sqrt(365) * 100:.1f}%")
        print(f"   Mean daily return: {self.return_mean * 100:.3f}%")
    
    async def _run_simulations(self):
        """Run Monte Carlo simulations"""
        simulation_results = []
        
        for sim_num in range(self.simulations):
            if sim_num % 100 == 0:
                print(f"   Simulation {sim_num + 1}/{self.simulations}")
            
            # Generate synthetic price path
            synthetic_prices = self._generate_synthetic_prices()
            
            # Run strategy on synthetic data
            strategy_result = self._run_strategy_simulation(synthetic_prices, sim_num)
            
            simulation_results.append(strategy_result)
        
        self.results = pd.DataFrame(simulation_results)
        print(f"✅ Completed {len(self.results)} simulations")
    
    def _generate_synthetic_prices(self):
        """Generate synthetic Bitcoin price path using Monte Carlo"""
        days = (self.end_date - self.start_date).days
        
        # Enhanced price simulation with volatility clustering
        returns = np.random.normal(self.return_mean, self.return_std, days)
        
        # Add volatility clustering (GARCH-like behavior)
        volatility = np.ones(days) * self.return_std
        for i in range(1, days):
            volatility[i] = 0.1 * self.return_std + 0.85 * volatility[i-1] + 0.05 * abs(returns[i-1])
            returns[i] = np.random.normal(self.return_mean, volatility[i])
        
        # Generate price path
        start_price = self.btc_data['Close'].iloc[0]
        prices = [start_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        return np.array(prices[1:])  # Skip initial price
    
    def _run_strategy_simulation(self, synthetic_prices, sim_num):
        """Run strategy on synthetic price data"""
        total_capital = 100000
        capital_deployed = 0
        total_btc = 0
        trades = 0
        last_trade_time = None
        
        # Calculate years outside the if block to ensure it's always available
        years = (self.end_date - self.start_date).days / 365.25
        
        # Calculate synthetic indicators
        price_series = pd.Series(synthetic_prices)
        highs = price_series.rolling(60, min_periods=1).max()
        drawdowns = (price_series - highs) / highs * 100
        
        # Simulate Fear & Greed (inversely correlated with drawdowns)
        fear_greed = np.clip(50 + drawdowns * 0.7, 0, 100)
        
        # Days since ATH calculation
        days_since_ath = np.zeros(len(synthetic_prices))
        for i in range(1, len(synthetic_prices)):
            if synthetic_prices[i] >= highs.iloc[i]:
                days_since_ath[i] = 0
            else:
                days_since_ath[i] = days_since_ath[i-1] + 1
        
        for day, (price, drawdown, fg, days_ath) in enumerate(zip(synthetic_prices, drawdowns, fear_greed, days_since_ath)):
            
            # Check cooldown (convert hours to days)
            if last_trade_time is not None:
                if day - last_trade_time < self.strategy_params['cooldown_hours'] / 24:
                    continue
            
            # Check entry criteria (more lenient for Monte Carlo)
            if (drawdown <= self.strategy_params['min_drawdown'] and  # Fixed: should be <=
                fg <= self.strategy_params['max_fear_greed'] and
                days_ath >= self.strategy_params['min_days_since_ath']):
                
                # Calculate Fibonacci levels (simplified)
                lookback_start = max(0, day - 60)
                recent_high = price_series.iloc[lookback_start:day+1].max()
                recent_low = price_series.iloc[lookback_start:day+1].min()
                price_range = recent_high - recent_low
                
                if price_range > recent_high * 0.05:  # More lenient range requirement
                    
                    # Find best Fibonacci opportunity
                    best_score = 0
                    best_level = None
                    
                    for level_name, level_config in self.strategy_params['fibonacci_levels'].items():
                        fib_price = recent_high - (price_range * level_config['ratio'])
                        distance_pct = (price - fib_price) / fib_price * 100
                        
                        # More lenient entry window for Monte Carlo
                        if -15.0 <= distance_pct <= -0.5:
                            leverage = min(
                                self.strategy_params['base_leverage'] + 
                                abs(drawdown) * 0.3 + 
                                (self.strategy_params['max_fear_greed'] - fg) * 0.2,
                                self.strategy_params['max_leverage']
                            )
                            
                            score = (level_config['base_multiplier'] * 
                                   leverage * 
                                   abs(distance_pct) * 
                                   (1 + abs(drawdown) / 100))
                            
                            if score > best_score:
                                best_score = score
                                best_level = {
                                    'multiplier': level_config['base_multiplier'],
                                    'leverage': leverage
                                }
                    
                    # Execute trade (lower threshold for Monte Carlo)
                    if best_level and best_score > 5:
                        remaining_capital = total_capital - capital_deployed
                        if remaining_capital > 100:  # Lower minimum
                            
                            base_position = min(remaining_capital * 0.15, 8000)  # More conservative
                            leverage = best_level['leverage']
                            total_position = base_position * leverage
                            
                            btc_acquired = total_position / price
                            
                            capital_deployed += base_position
                            total_btc += btc_acquired
                            trades += 1
                            last_trade_time = day
        
        # Calculate final performance
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
            
        else:
            total_return = annual_return = outperformance = 0
            buy_hold_annual = 0
            final_value = 0
        
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
            'trades_per_year': trades / years  # Now years is always defined
        }
    
    def _calculate_risk_metrics(self):
        """Calculate comprehensive risk metrics"""
        annual_returns = self.results['annual_return']
        
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
            
            # Drawdown Analysis
            'mean_max_drawdown': self.results['max_drawdown'].mean(),
            'worst_drawdown': self.results['max_drawdown'].min(),
            
            # Trading Statistics
            'mean_trades_per_year': self.results['trades_per_year'].mean(),
            'optimal_frequency': ((self.results['trades_per_year'] >= 15) & 
                                 (self.results['trades_per_year'] <= 20)).mean(),
            
            # Percentiles
            'return_percentiles': {
                '10th': np.percentile(annual_returns, 10),
                '25th': np.percentile(annual_returns, 25),
                '75th': np.percentile(annual_returns, 75),
                '90th': np.percentile(annual_returns, 90)
            }
        }
    
    def _stress_test_scenarios(self):
        """Generate stress test scenarios"""
        print(f"\n🔥 STRESS TEST SCENARIOS")
        
        # Define stress scenarios
        stress_scenarios = {
            'black_monday': {'return_shock': -0.20, 'volatility_multiplier': 3.0},
            'dot_com_crash': {'return_shock': -0.05, 'volatility_multiplier': 2.0, 'duration_days': 365},
            'financial_crisis': {'return_shock': -0.08, 'volatility_multiplier': 2.5, 'duration_days': 180},
            'flash_crash': {'return_shock': -0.35, 'volatility_multiplier': 5.0, 'duration_days': 1},
            'crypto_winter': {'return_shock': -0.03, 'volatility_multiplier': 1.8, 'duration_days': 730}
        }
        
        stress_results = {}
        
        for scenario_name, params in stress_scenarios.items():
            stressed_returns = []
            
            # Run 100 simulations per stress scenario
            for _ in range(100):
                days = (self.end_date - self.start_date).days
                returns = np.random.normal(self.return_mean, self.return_std, days)
                
                # Apply stress
                if 'duration_days' in params:
                    stress_start = np.random.randint(0, max(1, days - params['duration_days']))
                    stress_end = min(days, stress_start + params['duration_days'])
                    
                    for i in range(stress_start, stress_end):
                        returns[i] += params['return_shock'] / params['duration_days']
                        returns[i] = np.random.normal(returns[i], 
                                                    self.return_std * params['volatility_multiplier'])
                else:
                    # Single day shock
                    shock_day = np.random.randint(0, days)
                    returns[shock_day] += params['return_shock']
                
                # Calculate stressed performance
                stressed_prices = [self.btc_data['Close'].iloc[0]]
                for ret in returns:
                    stressed_prices.append(stressed_prices[-1] * (1 + ret))
                
                stressed_result = self._run_strategy_simulation(np.array(stressed_prices[1:]), 0)
                stressed_returns.append(stressed_result['annual_return'])
            
            stress_results[scenario_name] = {
                'mean_return': np.mean(stressed_returns),
                'min_return': np.min(stressed_returns),
                'var_95': np.percentile(stressed_returns, 5),
                'prob_positive': np.mean(np.array(stressed_returns) > 0)
            }
        
        self.stress_results = stress_results
        
        # Display stress test results
        for scenario, metrics in stress_results.items():
            print(f"\n   {scenario.replace('_', ' ').title()}:")
            print(f"      Mean Return: {metrics['mean_return']:+.1%}")
            print(f"      Worst Case: {metrics['min_return']:+.1%}")
            print(f"      VaR 95%: {metrics['var_95']:+.1%}")
            print(f"      Prob Positive: {metrics['prob_positive']:.1%}")
    
    def _display_results(self):
        """Display Monte Carlo results"""
        print(f"\n🎲 MONTE CARLO RISK ANALYSIS RESULTS")
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
        print(f"   Conditional VaR (99%): {self.risk_metrics['cvar_99']:+.1%}")
        
        print(f"\n📊 SUCCESS PROBABILITIES:")
        print(f"   Probability of Positive Returns: {self.risk_metrics['prob_positive']:.1%}")
        print(f"   Probability of Beating Target (+245.4%): {self.risk_metrics['prob_beat_target']:.1%}")
        print(f"   Probability of Beating Buy & Hold: {self.risk_metrics['prob_beat_buy_hold']:.1%}")
        print(f"   Probability of Optimal Trade Frequency: {self.risk_metrics['optimal_frequency']:.1%}")
        
        print(f"\n📊 DRAWDOWN ANALYSIS:")
        print(f"   Mean Maximum Drawdown: {self.risk_metrics['mean_max_drawdown']:.1%}")
        print(f"   Worst Drawdown: {self.risk_metrics['worst_drawdown']:.1%}")
        
        print(f"\n📊 TRADING CHARACTERISTICS:")
        print(f"   Mean Trades per Year: {self.risk_metrics['mean_trades_per_year']:.1f}")
        
        # Risk grade
        if (self.risk_metrics['prob_beat_target'] > 0.5 and 
            self.risk_metrics['sharpe_ratio'] > 2.0):
            grade = "A+ 🏆"
            assessment = "EXCELLENT RISK-ADJUSTED PERFORMANCE"
        elif (self.risk_metrics['prob_positive'] > 0.8 and 
              self.risk_metrics['prob_beat_buy_hold'] > 0.7):
            grade = "A 🎯"
            assessment = "STRONG RISK PROFILE"
        elif self.risk_metrics['prob_positive'] > 0.7:
            grade = "B+ 📊"
            assessment = "GOOD RISK MANAGEMENT"
        else:
            grade = "B 📈"
            assessment = "ACCEPTABLE RISK PROFILE"
        
        print(f"\n🏆 RISK GRADE: {grade}")
        print(f"💡 ASSESSMENT: {assessment}")
    
    def _create_visualizations(self):
        """Create risk analysis visualizations"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Monte Carlo Risk Analysis - Nanpin Strategy', fontsize=16, fontweight='bold')
            
            # 1. Return distribution histogram
            axes[0, 0].hist(self.results['annual_return'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            axes[0, 0].axvline(self.risk_metrics['mean_annual_return'], color='red', linestyle='--', 
                             label=f'Mean: {self.risk_metrics["mean_annual_return"]:.1%}')
            axes[0, 0].axvline(2.454, color='green', linestyle='--', label='Target: 245.4%')
            axes[0, 0].set_title('Annual Return Distribution')
            axes[0, 0].set_xlabel('Annual Return')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. VaR visualization
            var_data = [
                self.risk_metrics['var_95'],
                self.risk_metrics['var_99'],
                self.risk_metrics['cvar_95'],
                self.risk_metrics['cvar_99']
            ]
            var_labels = ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%']
            colors = ['orange', 'red', 'darkorange', 'darkred']
            
            bars = axes[0, 1].bar(var_labels, var_data, color=colors, alpha=0.7)
            axes[0, 1].set_title('Value at Risk Metrics')
            axes[0, 1].set_ylabel('Annual Return')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, var_data):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.1%}', ha='center', va='bottom')
            
            # 3. Trades per year vs Returns scatter (FIXED)
            # Filter out any invalid data points
            valid_mask = (self.results['trades_per_year'] >= 0) & (self.results['trades_per_year'] <= 100)
            valid_trades = self.results['trades_per_year'][valid_mask]
            valid_returns = self.results['annual_return'][valid_mask]
            
            axes[1, 0].scatter(valid_trades, valid_returns, 
                             alpha=0.6, s=30, color='purple')
            
            # Add optimal trade frequency band (15-20 trades per year)
            axes[1, 0].axvspan(15, 20, alpha=0.2, color='green', label='Optimal Trade Frequency')
            axes[1, 0].axhline(2.454, color='green', linestyle='--', label='Target Return (245.4%)')
            
            # Add mean trade frequency line
            mean_trades = valid_trades.mean()
            axes[1, 0].axvline(mean_trades, color='red', linestyle='--', alpha=0.7, 
                             label=f'Mean: {mean_trades:.1f} trades/year')
            
            axes[1, 0].set_title('Trade Frequency vs Returns')
            axes[1, 0].set_xlabel('Trades per Year')
            axes[1, 0].set_ylabel('Annual Return')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Set reasonable axis limits for better visualization
            if len(valid_trades) > 0:
                axes[1, 0].set_xlim(0, max(50, valid_trades.max() * 1.1))
            
            # 4. Cumulative probability of returns
            sorted_returns = np.sort(self.results['annual_return'])
            p = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
            axes[1, 1].plot(sorted_returns, p, linewidth=2, color='navy')
            axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.7, label='Break-even')
            axes[1, 1].axvline(2.454, color='green', linestyle='--', alpha=0.7, label='Target (245.4%)')
            axes[1, 1].set_title('Cumulative Probability Distribution')
            axes[1, 1].set_xlabel('Annual Return')
            axes[1, 1].set_ylabel('Cumulative Probability')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('monte_carlo_risk_analysis.png', dpi=300, bbox_inches='tight')
            print(f"\n📊 Visualizations saved to: monte_carlo_risk_analysis.png")
            
        except Exception as e:
            print(f"⚠️ Could not create visualizations: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Run Monte Carlo risk analysis"""
    print("🎲 MONTE CARLO RISK ANALYSIS - NANPIN STRATEGY")
    print("=" * 70)
    
    analyzer = MonteCarloRiskAnalyzer()
    results = await analyzer.run_monte_carlo_analysis()
    
    if results:
        print(f"\n🎉 MONTE CARLO ANALYSIS COMPLETE!")
        print(f"✅ Generated comprehensive risk assessment for Nanpin strategy")
        print(f"📊 Key insight: {results['prob_beat_target']:.1%} probability of hitting +245.4% target")

if __name__ == "__main__":
    asyncio.run(main())