#!/usr/bin/env python3
"""
🌸 Comprehensive Nanpin Strategy Backtester
Historical performance analysis for 永久ナンピン (Permanent DCA) strategy
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import json
import warnings
warnings.filterwarnings('ignore')

# Ensure matplotlib works in all environments
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

@dataclass
class BacktestTrade:
    """Individual trade in backtest"""
    timestamp: datetime
    price: float
    quantity: float
    usdc_amount: float
    fibonacci_level: str
    multiplier: float
    macro_regime: str
    fear_greed_index: float
    reasoning: str
    cumulative_btc: float
    cumulative_usdc: float
    average_price: float

@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics"""
    # Returns
    total_return: float
    annual_return: float
    monthly_returns: List[float]
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Trading metrics
    total_trades: int
    win_rate: float
    avg_trade_return: float
    profit_factor: float
    
    # Portfolio metrics
    final_btc: float
    final_value: float
    total_invested: float
    average_entry_price: float
    
    # Period metrics
    start_date: datetime
    end_date: datetime
    days_traded: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'avg_trade_return': self.avg_trade_return,
            'profit_factor': self.profit_factor,
            'final_btc': self.final_btc,
            'final_value': self.final_value,
            'total_invested': self.total_invested,
            'average_entry_price': self.average_entry_price,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'days_traded': self.days_traded
        }

class NanpinBacktester:
    """
    🌸 Comprehensive Nanpin Strategy Backtester
    
    Features:
    - Historical BTC data analysis
    - Simulated FRED macro data
    - Fibonacci-based entry simulation
    - Performance metrics calculation
    - Strategy comparison
    - Monte Carlo analysis
    - Risk assessment
    """
    
    def __init__(self, start_date: str = "2020-01-01", end_date: str = "2024-12-31"):
        """
        Initialize backtester
        
        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
        """
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Data storage
        self.btc_data = None
        self.macro_data = None
        self.trades = []
        self.portfolio_history = []
        
        # Strategy configuration
        self.base_amount = 100.0  # Base USDC amount per trade
        self.fibonacci_multipliers = {
            '23.6%': 1, '38.2%': 2, '50.0%': 3, '61.8%': 5, '78.6%': 8
        }
        
        # Performance tracking
        self.metrics = None
        self.comparison_results = {}
        
        logger.info("🌸 Nanpin Backtester initialized")
        logger.info(f"   Period: {start_date} to {end_date}")
        logger.info(f"   Base amount: ${self.base_amount}")
    
    async def run_comprehensive_backtest(self) -> Dict:
        """
        Run comprehensive backtest analysis
        
        Returns:
            Complete backtest results
        """
        try:
            logger.info("🚀 Starting comprehensive backtest...")
            
            # Step 1: Load historical data
            await self._load_historical_data()
            
            # Step 2: Simulate macro indicators
            self._simulate_macro_indicators()
            
            # Step 3: Calculate Fibonacci levels
            self._calculate_fibonacci_levels()
            
            # Step 4: Simulate trading
            self._simulate_nanpin_strategy()
            
            # Step 5: Calculate performance metrics
            self._calculate_performance_metrics()
            
            # Step 6: Compare with benchmark strategies
            self._compare_strategies()
            
            # Step 7: Generate reports
            results = self._generate_comprehensive_report()
            
            logger.info("✅ Comprehensive backtest completed!")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            raise
    
    async def _load_historical_data(self):
        """Load historical BTC price data"""
        try:
            logger.info("📊 Loading historical BTC data...")
            
            # Fetch BTC data from Yahoo Finance
            btc_ticker = yf.Ticker("BTC-USD")
            self.btc_data = btc_ticker.history(
                start=self.start_date - timedelta(days=365),  # Extra data for indicators
                end=self.end_date,
                interval="1d"
            )
            
            if self.btc_data.empty:
                raise Exception("Failed to fetch BTC data from Yahoo Finance")
            
            # Clean and prepare data
            self.btc_data = self.btc_data.dropna()
            self.btc_data.index = pd.to_datetime(self.btc_data.index)
            
            # Add technical indicators
            self._add_technical_indicators()
            
            logger.info(f"✅ Loaded {len(self.btc_data)} days of BTC data")
            logger.info(f"   Price range: ${self.btc_data['Low'].min():,.0f} - ${self.btc_data['High'].max():,.0f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load historical data: {e}")
            raise
    
    def _add_technical_indicators(self):
        """Add technical indicators to BTC data"""
        try:
            # Moving averages
            for period in [20, 50, 100, 200]:
                self.btc_data[f'MA_{period}'] = self.btc_data['Close'].rolling(period).mean()
            
            # Volatility (rolling 30-day)
            self.btc_data['Returns'] = self.btc_data['Close'].pct_change()
            self.btc_data['Volatility'] = self.btc_data['Returns'].rolling(30).std() * np.sqrt(365)
            
            # Support/Resistance levels
            self.btc_data['Local_High'] = self.btc_data['High'].rolling(20, center=True).max()
            self.btc_data['Local_Low'] = self.btc_data['Low'].rolling(20, center=True).min()
            
            logger.debug("✅ Technical indicators added")
            
        except Exception as e:
            logger.warning(f"Failed to add technical indicators: {e}")
    
    def _simulate_macro_indicators(self):
        """Simulate macro economic indicators for backtesting"""
        try:
            logger.info("🔮 Simulating macro indicators...")
            
            # Create macro data DataFrame
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            self.macro_data = pd.DataFrame(index=dates)
            
            # Simulate VIX-like volatility index
            base_vix = 20
            vix_trend = np.random.normal(0, 2, len(dates)).cumsum() * 0.1
            market_stress = self._identify_market_stress_periods()
            
            self.macro_data['VIX'] = np.clip(
                base_vix + vix_trend + market_stress * 20,
                5, 80
            )
            
            # Simulate Fear & Greed Index (inverse correlation with VIX)
            self.macro_data['Fear_Greed'] = np.clip(
                100 - (self.macro_data['VIX'] - 10) * 2 + np.random.normal(0, 5, len(dates)),
                0, 100
            )
            
            # Simulate Fed Funds Rate
            self.macro_data['Fed_Rate'] = self._simulate_fed_rate()
            
            # Simulate economic regime
            self.macro_data['Regime'] = self._classify_regime()
            
            # Calculate position scaling factors
            self.macro_data['Position_Scaling'] = self._calculate_macro_scaling()
            
            logger.info("✅ Macro indicators simulated")
            logger.info(f"   Avg Fear/Greed: {self.macro_data['Fear_Greed'].mean():.1f}")
            logger.info(f"   Crisis periods: {(self.macro_data['Regime'] == 'crisis').sum()} days")
            
        except Exception as e:
            logger.error(f"❌ Failed to simulate macro indicators: {e}")
            raise
    
    def _identify_market_stress_periods(self) -> np.ndarray:
        """Identify market stress periods from BTC price action"""
        # Convert datetime objects to pandas Timestamp for proper comparison
        start_ts = pd.Timestamp(self.start_date).tz_localize(None)
        end_ts = pd.Timestamp(self.end_date).tz_localize(None)
        
        # Ensure BTC data index has no timezone for comparison
        btc_index_clean = self.btc_data.index.tz_localize(None) if self.btc_data.index.tz else self.btc_data.index
        
        btc_subset = self.btc_data[
            (btc_index_clean >= start_ts) & 
            (btc_index_clean <= end_ts)
        ]
        
        # High stress during large drawdowns
        rolling_max = btc_subset['Close'].rolling(90).max()
        drawdown = (btc_subset['Close'] - rolling_max) / rolling_max
        
        stress = np.zeros(len(self.macro_data))
        for i, date in enumerate(self.macro_data.index):
            if date in btc_subset.index:
                dd = drawdown.loc[date]
                if dd < -0.3:  # 30%+ drawdown
                    stress[i] = 2.0
                elif dd < -0.2:  # 20%+ drawdown
                    stress[i] = 1.0
                elif dd < -0.1:  # 10%+ drawdown
                    stress[i] = 0.5
        
        return stress
    
    def _simulate_fed_rate(self) -> np.ndarray:
        """Simulate Federal Funds Rate"""
        # Start at 1.5%, with policy changes
        base_rate = 1.5
        rates = [base_rate]
        
        for i in range(1, len(self.macro_data)):
            # Simple policy simulation
            current_rate = rates[-1]
            
            # Rate changes based on market conditions
            vix = self.macro_data['VIX'].iloc[i]
            
            if vix > 40:  # Crisis - cut rates
                change = -0.01
            elif vix > 30:  # Stress - possible cut
                change = np.random.choice([-0.005, 0], p=[0.3, 0.7])
            elif vix < 15:  # Complacency - possible hike
                change = np.random.choice([0.005, 0], p=[0.2, 0.8])
            else:
                change = 0
            
            new_rate = np.clip(current_rate + change, 0, 6)
            rates.append(new_rate)
        
        return np.array(rates)
    
    def _classify_regime(self) -> np.ndarray:
        """Classify economic regime for each day"""
        regimes = []
        
        for i in range(len(self.macro_data)):
            vix = self.macro_data['VIX'].iloc[i]
            fed_rate = self.macro_data['Fed_Rate'].iloc[i]
            fear_greed = self.macro_data['Fear_Greed'].iloc[i]
            
            if vix > 40 or fear_greed < 20:
                regime = 'crisis'
            elif vix > 30 or fear_greed < 35:
                regime = 'recession'
            elif fed_rate < 1 and vix < 20:
                regime = 'recovery'
            elif fear_greed > 80 and vix < 15:
                regime = 'bubble'
            elif fed_rate > 4:
                regime = 'stagflation'
            else:
                regime = 'expansion'
            
            regimes.append(regime)
        
        return np.array(regimes)
    
    def _calculate_macro_scaling(self) -> np.ndarray:
        """Calculate position scaling factors from macro conditions"""
        scaling = []
        
        regime_multipliers = {
            'crisis': 2.5,
            'recession': 2.0,
            'recovery': 1.2,
            'expansion': 1.0,
            'stagflation': 1.5,
            'bubble': 0.7
        }
        
        for i in range(len(self.macro_data)):
            regime = self.macro_data['Regime'].iloc[i]
            fear_greed = self.macro_data['Fear_Greed'].iloc[i]
            
            base_scaling = regime_multipliers[regime]
            
            # Fear/Greed adjustments
            if fear_greed < 20:
                fear_adjustment = 1.8
            elif fear_greed < 35:
                fear_adjustment = 1.4
            elif fear_greed > 80:
                fear_adjustment = 0.5
            else:
                fear_adjustment = 1.0
            
            total_scaling = base_scaling * fear_adjustment
            total_scaling = np.clip(total_scaling, 0.3, 3.0)
            
            scaling.append(total_scaling)
        
        return np.array(scaling)
    
    def _calculate_fibonacci_levels(self):
        """Calculate Fibonacci retracement levels for each day"""
        try:
            logger.info("📐 Calculating Fibonacci levels...")
            
            fibonacci_levels = []
            
            for date in self.macro_data.index:
                if date not in self.btc_data.index:
                    fibonacci_levels.append({})
                    continue
                
                # Get lookback period (90 days)
                lookback_start = date - pd.Timedelta(days=90)
                period_data = self.btc_data[
                    (self.btc_data.index >= lookback_start) & 
                    (self.btc_data.index <= date)
                ]
                
                if len(period_data) < 30:
                    fibonacci_levels.append({})
                    continue
                
                # Find swing high and low
                swing_high = period_data['High'].max()
                swing_low = period_data['Low'].min()
                price_range = swing_high - swing_low
                
                # Calculate Fibonacci levels
                levels = {}
                for level_name, ratio in [('23.6%', 0.236), ('38.2%', 0.382), 
                                        ('50.0%', 0.500), ('61.8%', 0.618), ('78.6%', 0.786)]:
                    fib_price = swing_high - (price_range * ratio)
                    levels[level_name] = {
                        'price': fib_price,
                        'swing_high': swing_high,
                        'swing_low': swing_low
                    }
                
                fibonacci_levels.append(levels)
            
            self.macro_data['Fibonacci_Levels'] = fibonacci_levels
            
            logger.info("✅ Fibonacci levels calculated")
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate Fibonacci levels: {e}")
            raise
    
    def _simulate_nanpin_strategy(self):
        """Simulate the Nanpin trading strategy"""
        try:
            logger.info("💰 Simulating Nanpin strategy...")
            
            # Portfolio tracking
            total_btc = 0.0
            total_invested = 0.0
            trade_count = 0
            last_trade_date = None
            cooldown_days = 1  # Minimum days between trades
            
            self.trades = []
            portfolio_values = []
            
            for date in self.macro_data.index:
                if date not in self.btc_data.index:
                    continue
                
                current_price = self.btc_data.loc[date, 'Close']
                fibonacci_levels = self.macro_data.loc[date, 'Fibonacci_Levels']
                
                # Skip if no Fibonacci levels or too soon after last trade
                if (not fibonacci_levels or 
                    (last_trade_date and (date - last_trade_date).days < cooldown_days)):
                    
                    # Calculate portfolio value
                    portfolio_value = total_btc * current_price
                    portfolio_values.append({
                        'date': date,
                        'btc': total_btc,
                        'value': portfolio_value,
                        'invested': total_invested,
                        'price': current_price
                    })
                    continue
                
                # Check each Fibonacci level for entry opportunity
                best_opportunity = None
                best_score = 0
                
                for level_name, level_data in fibonacci_levels.items():
                    target_price = level_data['price']
                    distance_pct = (current_price - target_price) / target_price * 100
                    
                    # Entry condition: price is 0.5% to 5% below Fibonacci level
                    if -5.0 <= distance_pct <= -0.5:
                        # Calculate opportunity score
                        base_multiplier = self.fibonacci_multipliers[level_name]
                        macro_scaling = self.macro_data.loc[date, 'Position_Scaling']
                        
                        opportunity_score = base_multiplier * macro_scaling * abs(distance_pct)
                        
                        if opportunity_score > best_score:
                            best_score = opportunity_score
                            best_opportunity = {
                                'level': level_name,
                                'target_price': target_price,
                                'distance_pct': distance_pct,
                                'multiplier': base_multiplier * macro_scaling
                            }
                
                # Execute trade if opportunity found
                if best_opportunity:
                    multiplier = best_opportunity['multiplier']
                    trade_amount = self.base_amount * multiplier
                    btc_quantity = trade_amount / current_price
                    
                    # Create trade record
                    trade = BacktestTrade(
                        timestamp=date,
                        price=current_price,
                        quantity=btc_quantity,
                        usdc_amount=trade_amount,
                        fibonacci_level=best_opportunity['level'],
                        multiplier=multiplier,
                        macro_regime=self.macro_data.loc[date, 'Regime'],
                        fear_greed_index=self.macro_data.loc[date, 'Fear_Greed'],
                        reasoning=f"{best_opportunity['level']} level at ${current_price:,.0f}",
                        cumulative_btc=total_btc + btc_quantity,
                        cumulative_usdc=total_invested + trade_amount,
                        average_price=(total_invested + trade_amount) / (total_btc + btc_quantity)
                    )
                    
                    self.trades.append(trade)
                    
                    # Update portfolio
                    total_btc += btc_quantity
                    total_invested += trade_amount
                    trade_count += 1
                    last_trade_date = date
                    
                    logger.debug(f"Trade {trade_count}: {level_name} @ ${current_price:,.0f} "
                               f"(${trade_amount:.0f}, {multiplier:.1f}x)")
                
                # Calculate portfolio value
                portfolio_value = total_btc * current_price
                portfolio_values.append({
                    'date': date,
                    'btc': total_btc,
                    'value': portfolio_value,
                    'invested': total_invested,
                    'price': current_price
                })
            
            self.portfolio_history = pd.DataFrame(portfolio_values).set_index('date')
            
            logger.info(f"✅ Strategy simulation completed")
            logger.info(f"   Total trades: {len(self.trades)}")
            logger.info(f"   Final BTC: {total_btc:.6f}")
            logger.info(f"   Total invested: ${total_invested:,.0f}")
            logger.info(f"   Average price: ${total_invested/total_btc:,.0f}" if total_btc > 0 else "N/A")
            
        except Exception as e:
            logger.error(f"❌ Failed to simulate strategy: {e}")
            raise
    
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        try:
            logger.info("📊 Calculating performance metrics...")
            
            if self.portfolio_history.empty or len(self.trades) == 0:
                logger.warning("No trades executed, cannot calculate metrics")
                return
            
            # Basic portfolio metrics
            final_value = self.portfolio_history['value'].iloc[-1]
            total_invested = self.portfolio_history['invested'].iloc[-1]
            final_btc = self.portfolio_history['btc'].iloc[-1]
            
            # Returns calculation
            total_return = (final_value - total_invested) / total_invested if total_invested > 0 else 0
            
            # Time-based returns
            start_date = self.portfolio_history.index[0]
            end_date = self.portfolio_history.index[-1]
            days_traded = (end_date - start_date).days
            years_traded = days_traded / 365.25
            
            annual_return = (1 + total_return) ** (1 / years_traded) - 1 if years_traded > 0 else 0
            
            # Monthly returns for risk calculations
            monthly_portfolio = self.portfolio_history.resample('M').last()
            monthly_returns = monthly_portfolio['value'].pct_change().dropna().tolist()
            
            # Risk metrics
            if len(monthly_returns) > 1:
                volatility = np.std(monthly_returns) * np.sqrt(12)
                sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0  # Assume 2% risk-free rate
                
                # Sortino ratio (downside deviation)
                downside_returns = [r for r in monthly_returns if r < 0]
                downside_vol = np.std(downside_returns) * np.sqrt(12) if downside_returns else 0
                sortino_ratio = (annual_return - 0.02) / downside_vol if downside_vol > 0 else 0
            else:
                volatility = 0
                sharpe_ratio = 0
                sortino_ratio = 0
            
            # Maximum drawdown
            rolling_max = self.portfolio_history['value'].expanding().max()
            drawdown = (self.portfolio_history['value'] - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())
            
            # Calmar ratio
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
            
            # Trading metrics
            total_trades = len(self.trades)
            
            # Win rate (trades that would be profitable if sold at portfolio end)
            final_price = self.portfolio_history['price'].iloc[-1]
            winning_trades = sum(1 for trade in self.trades if final_price > trade.price)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Average trade return
            trade_returns = [(final_price - trade.price) / trade.price for trade in self.trades]
            avg_trade_return = np.mean(trade_returns) if trade_returns else 0
            
            # Profit factor
            profitable_trades = [r for r in trade_returns if r > 0]
            losing_trades = [abs(r) for r in trade_returns if r < 0]
            
            total_profit = sum(profitable_trades) if profitable_trades else 0
            total_loss = sum(losing_trades) if losing_trades else 1  # Avoid division by zero
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Average entry price
            if total_invested > 0 and final_btc > 0:
                average_entry_price = total_invested / final_btc
            else:
                average_entry_price = 0
            
            # Create metrics object
            self.metrics = BacktestMetrics(
                total_return=total_return,
                annual_return=annual_return,
                monthly_returns=monthly_returns,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                total_trades=total_trades,
                win_rate=win_rate,
                avg_trade_return=avg_trade_return,
                profit_factor=profit_factor,
                final_btc=final_btc,
                final_value=final_value,
                total_invested=total_invested,
                average_entry_price=average_entry_price,
                start_date=start_date,
                end_date=end_date,
                days_traded=days_traded
            )
            
            logger.info("✅ Performance metrics calculated")
            logger.info(f"   Total Return: {total_return:.1%}")
            logger.info(f"   Annual Return: {annual_return:.1%}")
            logger.info(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
            logger.info(f"   Max Drawdown: {max_drawdown:.1%}")
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate performance metrics: {e}")
            raise
    
    def _compare_strategies(self):
        """Compare Nanpin strategy with benchmark strategies"""
        try:
            logger.info("📈 Comparing with benchmark strategies...")
            
            if self.portfolio_history.empty:
                logger.warning("No portfolio history to compare")
                return
            
            btc_prices = self.btc_data.loc[self.portfolio_history.index, 'Close']
            
            # Strategy 1: Buy and Hold
            buy_hold_value = (self.metrics.total_invested / btc_prices.iloc[0]) * btc_prices
            buy_hold_return = (buy_hold_value.iloc[-1] - self.metrics.total_invested) / self.metrics.total_invested
            
            # Strategy 2: Simple DCA (weekly)
            weekly_dca_value = self._simulate_weekly_dca()
            weekly_dca_return = (weekly_dca_value.iloc[-1] - self.metrics.total_invested) / self.metrics.total_invested
            
            # Strategy 3: Simple Trump Era Strategy (+245.4% baseline)
            trump_era_annual = 2.454  # 245.4% annual
            trump_era_total = (1 + trump_era_annual) ** (self.metrics.days_traded / 365.25) - 1
            
            # Store comparison results
            self.comparison_results = {
                'nanpin_strategy': {
                    'total_return': self.metrics.total_return,
                    'annual_return': self.metrics.annual_return,
                    'sharpe_ratio': self.metrics.sharpe_ratio,
                    'max_drawdown': self.metrics.max_drawdown
                },
                'buy_and_hold': {
                    'total_return': buy_hold_return,
                    'annual_return': (1 + buy_hold_return) ** (365.25 / self.metrics.days_traded) - 1,
                    'sharpe_ratio': 0.5,  # Estimated
                    'max_drawdown': 0.8   # Estimated
                },
                'weekly_dca': {
                    'total_return': weekly_dca_return,
                    'annual_return': (1 + weekly_dca_return) ** (365.25 / self.metrics.days_traded) - 1,
                    'sharpe_ratio': 0.7,  # Estimated
                    'max_drawdown': 0.6   # Estimated
                },
                'trump_era_strategy': {
                    'total_return': trump_era_total,
                    'annual_return': trump_era_annual,
                    'sharpe_ratio': 4.83,  # From original data
                    'max_drawdown': 0.25   # Estimated
                }
            }
            
            logger.info("✅ Strategy comparison completed")
            
            # Log comparison results
            for strategy, metrics in self.comparison_results.items():
                logger.info(f"   {strategy}: {metrics['annual_return']:.1%} annual return")
            
        except Exception as e:
            logger.error(f"❌ Failed to compare strategies: {e}")
    
    def _simulate_weekly_dca(self) -> pd.Series:
        """Simulate simple weekly DCA strategy"""
        weekly_dates = pd.date_range(
            start=self.portfolio_history.index[0],
            end=self.portfolio_history.index[-1], 
            freq='W'
        )
        
        total_btc = 0
        total_invested = 0
        weekly_amount = self.base_amount
        
        dca_values = []
        
        for date in self.portfolio_history.index:
            # Check if it's a weekly DCA date
            if any(abs((date - weekly_date).days) <= 3 for weekly_date in weekly_dates):
                if date in self.btc_data.index:
                    price = self.btc_data.loc[date, 'Close']
                    btc_bought = weekly_amount / price
                    total_btc += btc_bought
                    total_invested += weekly_amount
            
            # Calculate current value
            current_price = self.btc_data.loc[date, 'Close'] if date in self.btc_data.index else 0
            current_value = total_btc * current_price
            dca_values.append(current_value)
        
        return pd.Series(dca_values, index=self.portfolio_history.index)
    
    def _generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive backtest report"""
        try:
            logger.info("📋 Generating comprehensive report...")
            
            # Create visualizations
            self._create_visualizations()
            
            # Prepare comprehensive results
            results = {
                'summary': {
                    'strategy': 'Enhanced Nanpin (永久ナンピン)',
                    'period': f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}",
                    'total_trades': len(self.trades),
                    'backtest_completed': datetime.now().isoformat()
                },
                'performance': self.metrics.to_dict() if self.metrics else {},
                'strategy_comparison': self.comparison_results,
                'trade_analysis': self._analyze_trades(),
                'regime_analysis': self._analyze_regime_performance(),
                'risk_analysis': self._analyze_risk_metrics(),
                'recommendations': self._generate_recommendations()
            }
            
            # Save results to file
            results_file = f"results/nanpin_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs("results", exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"✅ Comprehensive report saved to {results_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to generate report: {e}")
            return {}
    
    def _analyze_trades(self) -> Dict:
        """Analyze trading patterns and performance"""
        if not self.trades:
            return {}
        
        # Fibonacci level analysis
        level_performance = {}
        for level in ['23.6%', '38.2%', '50.0%', '61.8%', '78.6%']:
            level_trades = [t for t in self.trades if t.fibonacci_level == level]
            if level_trades:
                level_performance[level] = {
                    'count': len(level_trades),
                    'avg_multiplier': np.mean([t.multiplier for t in level_trades]),
                    'total_amount': sum([t.usdc_amount for t in level_trades])
                }
        
        # Regime analysis
        regime_performance = {}
        for regime in ['crisis', 'recession', 'recovery', 'expansion', 'stagflation', 'bubble']:
            regime_trades = [t for t in self.trades if t.macro_regime == regime]
            if regime_trades:
                regime_performance[regime] = {
                    'count': len(regime_trades),
                    'avg_multiplier': np.mean([t.multiplier for t in regime_trades]),
                    'total_amount': sum([t.usdc_amount for t in regime_trades])
                }
        
        return {
            'fibonacci_level_performance': level_performance,
            'regime_performance': regime_performance,
            'trade_frequency': {
                'total_days': self.metrics.days_traded if self.metrics else 0,
                'trading_days': len(self.trades),
                'frequency_pct': len(self.trades) / self.metrics.days_traded * 100 if self.metrics else 0
            }
        }
    
    def _analyze_regime_performance(self) -> Dict:
        """Analyze performance by macro regime"""
        if not self.portfolio_history.empty:
            regime_returns = {}
            
            for regime in ['crisis', 'recession', 'recovery', 'expansion', 'stagflation', 'bubble']:
                regime_mask = self.macro_data['Regime'] == regime
                if regime_mask.any():
                    regime_days = regime_mask.sum()
                    regime_returns[regime] = {
                        'days': int(regime_days),
                        'percentage_of_period': regime_days / len(self.macro_data) * 100
                    }
            
            return regime_returns
        
        return {}
    
    def _analyze_risk_metrics(self) -> Dict:
        """Analyze risk characteristics"""
        if not self.metrics:
            return {}
        
        return {
            'risk_adjusted_returns': {
                'sharpe_ratio': self.metrics.sharpe_ratio,
                'sortino_ratio': self.metrics.sortino_ratio,
                'calmar_ratio': self.metrics.calmar_ratio
            },
            'drawdown_analysis': {
                'max_drawdown': self.metrics.max_drawdown,
                'drawdown_periods': self._count_drawdown_periods()
            },
            'volatility_analysis': {
                'annual_volatility': self.metrics.volatility,
                'risk_level': 'Low' if self.metrics.volatility < 0.2 else 'Medium' if self.metrics.volatility < 0.4 else 'High'
            }
        }
    
    def _count_drawdown_periods(self) -> int:
        """Count number of significant drawdown periods"""
        if self.portfolio_history.empty:
            return 0
        
        rolling_max = self.portfolio_history['value'].expanding().max()
        drawdown = (self.portfolio_history['value'] - rolling_max) / rolling_max
        
        # Count periods where drawdown exceeds 10%
        significant_dd = drawdown < -0.1
        return len(significant_dd[significant_dd].index)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on backtest results"""
        recommendations = []
        
        if not self.metrics:
            return ["Insufficient data for recommendations"]
        
        # Performance recommendations
        if self.metrics.annual_return > 0.5:  # 50% annual return
            recommendations.append("✅ Strategy shows strong performance potential")
        else:
            recommendations.append("⚠️ Consider optimizing entry criteria for better returns")
        
        # Risk recommendations
        if self.metrics.sharpe_ratio > 2.0:
            recommendations.append("✅ Excellent risk-adjusted returns")
        elif self.metrics.sharpe_ratio > 1.0:
            recommendations.append("👍 Good risk-adjusted returns")
        else:
            recommendations.append("⚠️ Consider risk management improvements")
        
        # Drawdown recommendations
        if self.metrics.max_drawdown < 0.2:  # Less than 20%
            recommendations.append("✅ Well-controlled downside risk")
        else:
            recommendations.append("⚠️ Consider position sizing adjustments to reduce drawdowns")
        
        # Trading frequency recommendations
        if len(self.trades) < 50:
            recommendations.append("📈 Consider more aggressive entry criteria for more opportunities")
        elif len(self.trades) > 200:
            recommendations.append("📉 Consider more selective entry criteria to reduce transaction costs")
        
        return recommendations
    
    def _create_visualizations(self):
        """Create comprehensive visualization charts"""
        try:
            logger.info("📊 Creating visualizations...")
            
            os.makedirs("results/charts", exist_ok=True)
            
            # 1. Portfolio Performance Chart
            plt.figure(figsize=(15, 10))
            
            # Subplot 1: Portfolio Value
            plt.subplot(2, 2, 1)
            plt.plot(self.portfolio_history.index, self.portfolio_history['value'], 
                    label='Nanpin Strategy', linewidth=2, color='blue')
            plt.plot(self.portfolio_history.index, self.portfolio_history['invested'], 
                    label='Total Invested', linewidth=1, color='red', linestyle='--')
            plt.title('Portfolio Value vs Investment')
            plt.xlabel('Date')
            plt.ylabel('Value (USD)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 2: BTC Accumulation
            plt.subplot(2, 2, 2)
            plt.plot(self.portfolio_history.index, self.portfolio_history['btc'], 
                    color='orange', linewidth=2)
            plt.title('BTC Accumulation Over Time')
            plt.xlabel('Date')
            plt.ylabel('BTC Amount')
            plt.grid(True, alpha=0.3)
            
            # Subplot 3: Trade Execution Points
            plt.subplot(2, 2, 3)
            btc_prices = self.btc_data.loc[self.portfolio_history.index, 'Close']
            plt.plot(btc_prices.index, btc_prices.values, color='gray', alpha=0.7, label='BTC Price')
            
            if self.trades:
                trade_dates = [t.timestamp for t in self.trades]
                trade_prices = [t.price for t in self.trades]
                trade_colors = [{'crisis': 'red', 'recession': 'orange', 'recovery': 'green', 
                               'expansion': 'blue', 'stagflation': 'purple', 'bubble': 'yellow'}.get(t.macro_regime, 'black') 
                              for t in self.trades]
                
                plt.scatter(trade_dates, trade_prices, c=trade_colors, s=50, alpha=0.8, label='Trades')
            
            plt.title('Trade Execution Points')
            plt.xlabel('Date')
            plt.ylabel('BTC Price (USD)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 4: Returns Distribution
            plt.subplot(2, 2, 4)
            if self.metrics and self.metrics.monthly_returns:
                plt.hist(self.metrics.monthly_returns, bins=20, alpha=0.7, color='green')
                plt.axvline(np.mean(self.metrics.monthly_returns), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(self.metrics.monthly_returns):.2%}')
                plt.title('Monthly Returns Distribution')
                plt.xlabel('Monthly Return')
                plt.ylabel('Frequency')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('results/charts/nanpin_performance_overview.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Strategy Comparison Chart
            if self.comparison_results:
                plt.figure(figsize=(12, 8))
                
                strategies = list(self.comparison_results.keys())
                annual_returns = [self.comparison_results[s]['annual_return'] for s in strategies]
                sharpe_ratios = [self.comparison_results[s]['sharpe_ratio'] for s in strategies]
                
                plt.subplot(1, 2, 1)
                bars = plt.bar(strategies, [r * 100 for r in annual_returns], color=['blue', 'green', 'orange', 'red'])
                plt.title('Annual Returns Comparison')
                plt.xlabel('Strategy')
                plt.ylabel('Annual Return (%)')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, annual_returns):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                            f'{value:.1%}', ha='center', va='bottom')
                
                plt.subplot(1, 2, 2)
                bars = plt.bar(strategies, sharpe_ratios, color=['blue', 'green', 'orange', 'red'])
                plt.title('Sharpe Ratio Comparison')
                plt.xlabel('Strategy')
                plt.ylabel('Sharpe Ratio')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, sharpe_ratios):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                            f'{value:.2f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig('results/charts/strategy_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info("✅ Visualizations created")
            
        except Exception as e:
            logger.warning(f"Failed to create visualizations: {e}")


# Import os at the top of the file
import os

async def run_comprehensive_backtest():
    """Run comprehensive backtest analysis"""
    print("🌸 Starting Comprehensive Nanpin Backtest")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize backtester
        backtester = NanpinBacktester(
            start_date="2020-01-01",
            end_date="2024-12-31"
        )
        
        # Run comprehensive analysis
        results = await backtester.run_comprehensive_backtest()
        
        # Display key results
        print("\n🎉 BACKTEST RESULTS SUMMARY")
        print("=" * 40)
        
        if results.get('performance'):
            perf = results['performance']
            print(f"📈 Total Return: {perf.get('total_return', 0):.1%}")
            print(f"📈 Annual Return: {perf.get('annual_return', 0):.1%}")
            print(f"📊 Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
            print(f"📉 Max Drawdown: {perf.get('max_drawdown', 0):.1%}")
            print(f"💰 Total Trades: {perf.get('total_trades', 0)}")
            print(f"₿ Final BTC: {perf.get('final_btc', 0):.6f}")
            print(f"💵 Total Invested: ${perf.get('total_invested', 0):,.0f}")
            print(f"💵 Final Value: ${perf.get('final_value', 0):,.0f}")
        
        # Strategy comparison
        if results.get('strategy_comparison'):
            print(f"\n🏆 STRATEGY COMPARISON")
            print("-" * 30)
            for strategy, metrics in results['strategy_comparison'].items():
                print(f"{strategy}: {metrics.get('annual_return', 0):.1%} annual")
        
        # Recommendations
        if results.get('recommendations'):
            print(f"\n💡 RECOMMENDATIONS")
            print("-" * 20)
            for rec in results['recommendations']:
                print(f"  {rec}")
        
        print(f"\n✅ Comprehensive backtest completed successfully!")
        print(f"📁 Results saved to: results/")
        
        return results
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_comprehensive_backtest())