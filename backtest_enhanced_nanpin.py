#!/usr/bin/env python3
"""
🧪 Enhanced Nanpin Bot Backtesting System
Comprehensive backtesting to verify Sharpe ratio and performance metrics
"""

import asyncio
import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import yaml
import os
from pathlib import Path

# Import our strategy components
import sys
sys.path.append('src')

@dataclass
class BacktestResult:
    """Backtest results container"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    avg_trade_duration: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    daily_returns: List[float]
    equity_curve: List[float]
    trade_log: List[Dict]

class EnhancedNanpinBacktester:
    """
    🧪 Enhanced Nanpin Strategy Backtester
    
    Tests the enhanced strategy with:
    - Fibonacci-based entries
    - Macro regime detection
    - Liquidation intelligence
    - Dynamic position sizing
    """
    
    def __init__(self, config_path: str = "config/enhanced_nanpin_config.yaml"):
        """Initialize backtester with enhanced configuration"""
        
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Strategy parameters from config
        strategy_config = self.config['nanpin_strategy']
        self.base_investment = strategy_config['base_investment']
        self.scaling_multiplier = strategy_config['scaling_multiplier']
        self.price_drop_threshold = strategy_config['price_drop_threshold']
        self.take_profit_percentage = strategy_config['take_profit_percentage']
        self.max_drawdown_stop = strategy_config['max_drawdown_stop']
        self.max_nanpin_levels = self.config['trading']['max_nanpin_levels']
        
        # Fibonacci levels
        self.fibonacci_levels = strategy_config['fibonacci_levels']
        self.entry_windows = strategy_config['entry_windows']
        
        # Risk management
        self.min_confidence_threshold = self.config['trading']['min_confidence_threshold']
        self.max_position_size = self.config['trading']['max_position_size']
        
        # Performance targets for comparison
        self.targets = self.config['performance_targets']
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🧪 Enhanced Nanpin Backtester initialized")
        self.logger.info(f"   Target Sharpe Ratio: {self.targets['sharpe_ratio_target']}")
        self.logger.info(f"   Target Max Drawdown: {self.targets['max_drawdown']*100}%")
        self.logger.info(f"   Target Win Rate: {self.targets['win_rate_target']*100}%")
    
    def fetch_historical_data(self, symbol: str = "BTC-USD", period: str = "2y") -> pd.DataFrame:
        """Fetch historical price data"""
        try:
            self.logger.info(f"📈 Fetching {period} of historical data for {symbol}...")
            
            # Use yfinance for reliable BTC data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval="1h")
            
            if data.empty:
                raise ValueError(f"No data retrieved for {symbol}")
            
            # Add technical indicators
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(24).std() * np.sqrt(24)  # 24-hour volatility
            data['SMA_50'] = data['Close'].rolling(50).mean()
            data['SMA_200'] = data['Close'].rolling(200).mean()
            
            # Add macro regime proxy (simplified)
            data['Trend'] = np.where(data['SMA_50'] > data['SMA_200'], 'bullish', 'bearish')
            data['Regime'] = np.where(data['Volatility'] > data['Volatility'].rolling(100).mean(), 'high_vol', 'low_vol')
            
            self.logger.info(f"   ✅ Retrieved {len(data)} data points")
            self.logger.info(f"   Date range: {data.index[0]} to {data.index[-1]}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching historical data: {e}")
            raise
    
    def calculate_fibonacci_levels(self, high: float, low: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        try:
            diff = high - low
            levels = {}
            
            for level_name, level_data in self.fibonacci_levels.items():
                ratio = level_data['ratio']
                levels[level_name] = high - (diff * ratio)
            
            return levels
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating Fibonacci levels: {e}")
            return {}
    
    def simulate_macro_regime(self, row: pd.Series) -> Dict:
        """Simulate macro regime detection"""
        try:
            # Simplified macro regime simulation
            volatility = row.get('Volatility', 0.02)
            trend = row.get('Trend', 'neutral')
            
            if volatility < 0.015 and trend == 'bullish':
                regime = 'expansion'
                confidence = 0.8
            elif volatility > 0.04:
                regime = 'crisis'
                confidence = 0.9
            elif trend == 'bearish':
                regime = 'contraction'
                confidence = 0.7
            else:
                regime = 'neutral'
                confidence = 0.5
            
            return {
                'regime': regime,
                'confidence': confidence,
                'signal': 'bullish' if regime == 'expansion' else 'bearish' if regime == 'crisis' else 'neutral'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error simulating macro regime: {e}")
            return {'regime': 'neutral', 'confidence': 0.5, 'signal': 'neutral'}
    
    def evaluate_entry_opportunity(self, current_price: float, recent_high: float, recent_low: float, macro_data: Dict) -> Optional[Dict]:
        """Evaluate if current price presents a good entry opportunity"""
        try:
            # Calculate Fibonacci levels from recent high/low
            fib_levels = self.calculate_fibonacci_levels(recent_high, recent_low)
            
            best_opportunity = None
            best_score = 0
            
            for level_name, fib_price in fib_levels.items():
                level_config = self.fibonacci_levels[level_name]
                entry_window = self.entry_windows.get(level_name, [-5.0, -0.5])
                
                # Calculate distance from Fibonacci level
                distance_pct = ((current_price - fib_price) / fib_price) * 100
                
                # Check if within entry window
                if entry_window[0] <= distance_pct <= entry_window[1]:
                    # Calculate opportunity score
                    base_confidence = level_config['confidence']
                    macro_boost = macro_data['confidence'] * 0.3 if macro_data['signal'] == 'bullish' else 0
                    
                    opportunity_score = (base_confidence + macro_boost) * 100
                    
                    # Adjust for market regime
                    if macro_data['regime'] == 'expansion':
                        opportunity_score *= 1.2
                    elif macro_data['regime'] == 'crisis':
                        opportunity_score *= 0.7
                    
                    if opportunity_score > best_score and opportunity_score >= (self.min_confidence_threshold * 100):
                        best_opportunity = {
                            'level': level_name,
                            'entry_price': current_price,
                            'target_price': fib_price,
                            'confidence': opportunity_score / 100,
                            'position_size': level_config['base_multiplier'] * self.base_investment,
                            'leverage': min(level_config['base_multiplier'], 10),
                            'macro_regime': macro_data['regime']
                        }
                        best_score = opportunity_score
            
            return best_opportunity
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluating entry opportunity: {e}")
            return None
    
    def run_backtest(self, start_date: str = None, end_date: str = None, initial_capital: float = 10000) -> BacktestResult:
        """Run comprehensive backtest"""
        try:
            self.logger.info(f"🚀 Starting Enhanced Nanpin Backtest...")
            self.logger.info(f"   Initial Capital: ${initial_capital:,.2f}")
            
            # Fetch historical data
            data = self.fetch_historical_data()
            
            # Filter date range if specified
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            # Initialize backtest variables
            capital = initial_capital
            positions = []
            trade_log = []
            equity_curve = [initial_capital]
            daily_returns = []
            
            # Lookback periods for Fibonacci calculations
            lookback_high = 168  # 1 week in hours
            lookback_low = 168
            
            self.logger.info(f"   Backtesting period: {data.index[0]} to {data.index[-1]}")
            self.logger.info(f"   Total data points: {len(data)}")
            
            for i, (timestamp, row) in enumerate(data.iterrows()):
                if i < max(lookback_high, lookback_low, 200):  # Skip initial period for indicators
                    continue
                
                current_price = row['Close']
                
                # Calculate recent high/low for Fibonacci levels
                recent_data = data.iloc[i-lookback_high:i]
                recent_high = recent_data['High'].max()
                recent_low = recent_data['Low'].min()
                
                # Simulate macro regime
                macro_data = self.simulate_macro_regime(row)
                
                # Check for exit conditions on existing positions
                positions_to_close = []
                for pos_idx, position in enumerate(positions):
                    # Take profit check
                    if current_price >= position['entry_price'] * (1 + self.take_profit_percentage):
                        profit = (current_price - position['entry_price']) * position['size'] / position['entry_price']
                        capital += position['size'] + profit
                        
                        trade_log.append({
                            'timestamp': timestamp,
                            'action': 'SELL',
                            'price': current_price,
                            'size': position['size'],
                            'profit': profit,
                            'level': position['level'],
                            'duration': (timestamp - position['timestamp']).total_seconds() / 3600,  # hours
                            'reason': 'take_profit'
                        })
                        
                        positions_to_close.append(pos_idx)
                    
                    # Stop loss check
                    elif current_price <= position['entry_price'] * (1 - self.max_drawdown_stop):
                        loss = (position['entry_price'] - current_price) * position['size'] / position['entry_price']
                        capital += position['size'] - loss
                        
                        trade_log.append({
                            'timestamp': timestamp,
                            'action': 'SELL',
                            'price': current_price,
                            'size': position['size'],
                            'profit': -loss,
                            'level': position['level'],
                            'duration': (timestamp - position['timestamp']).total_seconds() / 3600,
                            'reason': 'stop_loss'
                        })
                        
                        positions_to_close.append(pos_idx)
                
                # Remove closed positions
                for pos_idx in reversed(positions_to_close):
                    positions.pop(pos_idx)
                
                # Check for new entry opportunities
                if len(positions) < self.max_nanpin_levels:  # Don't exceed max levels
                    opportunity = self.evaluate_entry_opportunity(current_price, recent_high, recent_low, macro_data)
                    
                    if opportunity and capital >= opportunity['position_size']:
                        # Open new position
                        capital -= opportunity['position_size']
                        
                        positions.append({
                            'timestamp': timestamp,
                            'entry_price': current_price,
                            'size': opportunity['position_size'],
                            'level': opportunity['level'],
                            'confidence': opportunity['confidence'],
                            'macro_regime': opportunity['macro_regime']
                        })
                        
                        trade_log.append({
                            'timestamp': timestamp,
                            'action': 'BUY',
                            'price': current_price,
                            'size': opportunity['position_size'],
                            'level': opportunity['level'],
                            'confidence': opportunity['confidence'],
                            'macro_regime': opportunity['macro_regime'],
                            'reason': 'fibonacci_entry'
                        })
                
                # Calculate current portfolio value
                position_value = sum([pos['size'] * current_price / pos['entry_price'] for pos in positions])
                total_value = capital + position_value
                equity_curve.append(total_value)
                
                # Calculate daily returns (using hourly data, so convert)
                if len(equity_curve) > 24:  # 24 hours
                    daily_return = (equity_curve[-1] / equity_curve[-25]) - 1  # 24 hours ago
                    daily_returns.append(daily_return)
            
            # Calculate final metrics
            total_return = (equity_curve[-1] / initial_capital) - 1
            
            # Convert to numpy for calculations
            returns_array = np.array(daily_returns)
            equity_array = np.array(equity_curve)
            
            # Sharpe Ratio (annualized)
            if len(returns_array) > 0 and np.std(returns_array) > 0:
                sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(365)
            else:
                sharpe_ratio = 0
            
            # Maximum Drawdown
            peak = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - peak) / peak
            max_drawdown = abs(np.min(drawdown))
            
            # Win Rate
            profitable_trades = [t for t in trade_log if t.get('profit', 0) > 0]
            total_trades = len([t for t in trade_log if 'profit' in t])
            win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0
            
            # Average Trade Duration
            trade_durations = [t['duration'] for t in trade_log if 'duration' in t]
            avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
            
            # Volatility (annualized)
            volatility = np.std(returns_array) * np.sqrt(365) if len(returns_array) > 0 else 0
            
            # Calmar Ratio
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
            
            # Sortino Ratio
            negative_returns = returns_array[returns_array < 0]
            downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0
            sortino_ratio = np.mean(returns_array) / downside_deviation * np.sqrt(365) if downside_deviation > 0 else 0
            
            result = BacktestResult(
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                num_trades=total_trades,
                avg_trade_duration=avg_trade_duration,
                volatility=volatility,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                daily_returns=daily_returns,
                equity_curve=equity_curve,
                trade_log=trade_log
            )
            
            self.logger.info("✅ Backtest completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Backtest failed: {e}")
            raise
    
    def generate_performance_report(self, result: BacktestResult) -> str:
        """Generate comprehensive performance report"""
        try:
            report = []
            report.append("🧪 ENHANCED NANPIN BACKTEST RESULTS")
            report.append("=" * 50)
            report.append("")
            
            # Performance Metrics
            report.append("📊 PERFORMANCE METRICS")
            report.append("-" * 25)
            report.append(f"Total Return:        {result.total_return*100:>8.2f}%")
            report.append(f"Sharpe Ratio:        {result.sharpe_ratio:>8.2f}")
            report.append(f"Sortino Ratio:       {result.sortino_ratio:>8.2f}")
            report.append(f"Calmar Ratio:        {result.calmar_ratio:>8.2f}")
            report.append(f"Max Drawdown:        {result.max_drawdown*100:>8.2f}%")
            report.append(f"Volatility:          {result.volatility*100:>8.2f}%")
            report.append("")
            
            # Trading Metrics
            report.append("📈 TRADING METRICS")
            report.append("-" * 25)
            report.append(f"Total Trades:        {result.num_trades:>8d}")
            report.append(f"Win Rate:            {result.win_rate*100:>8.2f}%")
            report.append(f"Avg Trade Duration:  {result.avg_trade_duration:>8.1f} hrs")
            report.append("")
            
            # Target Comparison
            report.append("🎯 TARGET COMPARISON")
            report.append("-" * 25)
            
            # Sharpe Ratio comparison
            target_sharpe = self.targets['sharpe_ratio_target']
            sharpe_status = "✅ PASS" if result.sharpe_ratio >= target_sharpe else "❌ FAIL"
            report.append(f"Sharpe Ratio:        {result.sharpe_ratio:>8.2f} (Target: {target_sharpe:.1f}) {sharpe_status}")
            
            # Max Drawdown comparison
            target_drawdown = self.targets['max_drawdown']
            drawdown_status = "✅ PASS" if result.max_drawdown <= target_drawdown else "❌ FAIL"
            report.append(f"Max Drawdown:        {result.max_drawdown*100:>8.2f}% (Target: {target_drawdown*100:.1f}%) {drawdown_status}")
            
            # Win Rate comparison
            target_winrate = self.targets['win_rate_target']
            winrate_status = "✅ PASS" if result.win_rate >= target_winrate else "❌ FAIL"
            report.append(f"Win Rate:            {result.win_rate*100:>8.2f}% (Target: {target_winrate*100:.1f}%) {winrate_status}")
            
            report.append("")
            
            # Overall Assessment
            passes = sum([
                result.sharpe_ratio >= target_sharpe,
                result.max_drawdown <= target_drawdown,
                result.win_rate >= target_winrate
            ])
            
            if passes == 3:
                assessment = "🎉 EXCELLENT - All targets met!"
            elif passes == 2:
                assessment = "👍 GOOD - Most targets met"
            elif passes == 1:
                assessment = "⚠️ NEEDS IMPROVEMENT - Few targets met"
            else:
                assessment = "❌ POOR - Targets not met"
            
            report.append(f"OVERALL ASSESSMENT: {assessment}")
            report.append("")
            
            # Recent Trades Sample
            if result.trade_log:
                report.append("📝 RECENT TRADES (Last 5)")
                report.append("-" * 25)
                recent_trades = result.trade_log[-5:]
                for trade in recent_trades:
                    profit_str = f"{trade.get('profit', 0):+.2f}" if 'profit' in trade else "N/A"
                    report.append(f"{trade['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
                                f"{trade['action']:>4} | "
                                f"${trade['price']:>8.2f} | "
                                f"Profit: ${profit_str:>8} | "
                                f"Level: {trade.get('level', 'N/A')}")
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating report: {e}")
            return f"Error generating report: {e}"

async def main():
    """Main backtesting function"""
    try:
        print("🧪 Starting Enhanced Nanpin Backtest System...")
        
        # Initialize backtester
        backtester = EnhancedNanpinBacktester()
        
        # Run backtest
        print("🚀 Running backtest...")
        result = backtester.run_backtest(initial_capital=10000)
        
        # Generate and display report
        report = backtester.generate_performance_report(result)
        print("\n" + report)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trade log
        trade_log_path = f"backtest_results/trades_{timestamp}.json"
        os.makedirs("backtest_results", exist_ok=True)
        
        with open(trade_log_path, 'w') as f:
            # Convert timestamps to strings for JSON serialization
            trade_log_serializable = []
            for trade in result.trade_log:
                trade_copy = trade.copy()
                if 'timestamp' in trade_copy:
                    trade_copy['timestamp'] = trade_copy['timestamp'].isoformat()
                trade_log_serializable.append(trade_copy)
            
            json.dump(trade_log_serializable, f, indent=2)
        
        # Save equity curve
        equity_df = pd.DataFrame({
            'equity': result.equity_curve,
            'returns': [0] + result.daily_returns
        })
        equity_df.to_csv(f"backtest_results/equity_curve_{timestamp}.csv", index=False)
        
        print(f"\n💾 Detailed results saved:")
        print(f"   Trade log: {trade_log_path}")
        print(f"   Equity curve: backtest_results/equity_curve_{timestamp}.csv")
        
        return result
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        raise

if __name__ == "__main__":
    # Install required packages if needed
    try:
        import yfinance
    except ImportError:
        print("Installing yfinance...")
        os.system("pip install yfinance")
        import yfinance
    
    # Run backtest
    asyncio.run(main())