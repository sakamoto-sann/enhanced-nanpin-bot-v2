#!/usr/bin/env python3
"""
🎯 Goldilocks PLUS Nanpin Strategy
More aggressive version to achieve 15-20 trades/year target
"""

import asyncio
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class GoldilocksPlusNanpin:
    """More aggressive Goldilocks strategy to hit trade frequency targets"""
    
    def __init__(self, start_date="2020-01-01", end_date="2024-12-31"):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.total_capital = 100000  # $100K total
        
        # GOLDILOCKS PLUS PARAMETERS (more aggressive)
        self.optimal_criteria = {
            # More aggressive entry criteria
            'min_drawdown': -18,       # Reduced from -22% to capture more opportunities
            'max_fear_greed': 35,      # Increased from 25 to capture medium fear periods
            'min_days_since_ath': 7,   # Reduced from 14 days
            
            # Include 23.6% level but with lower multiplier
            'fibonacci_levels': {
                '23.6%': {'ratio': 0.236, 'base_multiplier': 2},   # Light retracement
                '38.2%': {'ratio': 0.382, 'base_multiplier': 3},   # Medium retracement
                '50.0%': {'ratio': 0.500, 'base_multiplier': 5},   # Major retracement  
                '61.8%': {'ratio': 0.618, 'base_multiplier': 8},   # Golden ratio
                '78.6%': {'ratio': 0.786, 'base_multiplier': 13}   # Deep retracement
            },
            
            # More aggressive leverage
            'leverage_scaling': {
                'base_leverage': 3.0,       # Increased from 2x to 3x
                'max_leverage': 18.0,       # Increased from 15x to 18x
                'drawdown_multiplier': 0.6,  # Increased sensitivity
                'fear_multiplier': 0.4      # Increased sensitivity
            },
            
            # Reduced cooldown for more trades
            'cooldown_hours': 48,          # Reduced from 72 hours
            'dynamic_cooldown': True
        }
        
        # More specific entry windows
        self.entry_windows = {
            '23.6%': (-3.0, -0.5),   # 0.5% to 3% below
            '38.2%': (-5.0, -1.0),   # 1% to 5% below
            '50.0%': (-7.0, -1.5),   # 1.5% to 7% below
            '61.8%': (-10.0, -2.0),  # 2% to 10% below
            '78.6%': (-15.0, -3.0)   # 3% to 15% below
        }
        
        self.trades = []
        self.capital_deployed = 0
        
    async def run_goldilocks_plus_backtest(self):
        """Run Goldilocks Plus backtest"""
        try:
            print("🎯 GOLDILOCKS PLUS NANPIN STRATEGY")
            print("="*58)
            print("Strategy: More aggressive balance to hit 15-20 trades/year")
            print(f"Capital: ${self.total_capital:,}")
            print(f"Criteria: -18% drawdown, F&G ≤35, include 23.6% level")
            print(f"Cooldown: 48 hours, higher leverage (3-18x)")
            
            # Load data
            await self._load_data()
            
            # Calculate opportunities
            self._calculate_opportunities()
            
            # Execute strategy
            self._execute_strategy()
            
            # Calculate performance
            results = self._calculate_performance()
            
            # Display results
            self._display_results(results)
            
            return results
            
        except Exception as e:
            print(f"❌ Goldilocks Plus failed: {e}")
            return {}
    
    async def _load_data(self):
        """Load and prepare data"""
        btc = yf.Ticker("BTC-USD")
        data = btc.history(
            start=(self.start_date - timedelta(days=150)).strftime("%Y-%m-%d"),
            end=self.end_date.strftime("%Y-%m-%d"),
            interval="1d"
        )
        
        self.btc_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        self.btc_data.index = pd.to_datetime(self.btc_data.index.date)
        self.btc_data = self.btc_data.dropna()
        
        # Add indicators
        self._add_indicators()
        
        print(f"✅ Data loaded: {len(self.btc_data)} days")
    
    def _add_indicators(self):
        """Add technical indicators"""
        # Rolling ATH (shorter 60-day window for more responsive signals)
        self.btc_data['ATH_60d'] = self.btc_data['High'].rolling(60, min_periods=1).max()
        
        # Drawdown from 60-day ATH
        self.btc_data['Drawdown'] = (self.btc_data['Close'] - self.btc_data['ATH_60d']) / self.btc_data['ATH_60d'] * 100
        
        # Days since ATH (simplified calculation)
        self.btc_data['Days_Since_ATH'] = 0
        for i in range(1, len(self.btc_data)):
            if self.btc_data.iloc[i]['High'] >= self.btc_data.iloc[i]['ATH_60d']:
                self.btc_data.iloc[i, self.btc_data.columns.get_loc('Days_Since_ATH')] = 0
            else:
                self.btc_data.iloc[i, self.btc_data.columns.get_loc('Days_Since_ATH')] = self.btc_data.iloc[i-1]['Days_Since_ATH'] + 1
        
        # Enhanced Fear & Greed
        returns = self.btc_data['Close'].pct_change()
        volatility = returns.rolling(10).std() * np.sqrt(365) * 100
        momentum_3d = (self.btc_data['Close'] / self.btc_data['Close'].shift(3) - 1) * 100
        
        self.btc_data['Fear_Greed'] = np.clip(
            50 + 
            self.btc_data['Drawdown'] * 0.7 +      # Stronger drawdown impact
            momentum_3d * 0.4 -                    # 3-day momentum
            (volatility - 40) * 0.3,               # Volatility impact
            0, 100
        )
        
        self.btc_data['Volatility'] = volatility
    
    def _calculate_opportunities(self):
        """Calculate trading opportunities"""
        backtest_data = self.btc_data[
            (self.btc_data.index >= self.start_date) &
            (self.btc_data.index <= self.end_date)
        ].copy()
        
        opportunities = []
        
        for date in backtest_data.index:
            # Shorter lookback for more responsive signals
            volatility = backtest_data.loc[date, 'Volatility']
            if pd.notna(volatility) and volatility > 70:
                lookback_days = 45
            elif pd.notna(volatility) and volatility < 30:
                lookback_days = 75
            else:
                lookback_days = 60
            
            lookback_start = date - pd.Timedelta(days=lookback_days)
            period_data = self.btc_data[
                (self.btc_data.index >= lookback_start) & 
                (self.btc_data.index <= date)
            ]
            
            if len(period_data) < 20:  # Reduced minimum
                opportunities.append({})
                continue
            
            # Calculate swing points
            swing_high = period_data['High'].max()
            swing_low = period_data['Low'].min()
            price_range = swing_high - swing_low
            
            if price_range < swing_high * 0.1:
                opportunities.append({})
                continue
            
            # Calculate ALL Fibonacci levels (including 23.6%)
            day_opportunities = {}
            current_price = backtest_data.loc[date, 'Close']
            
            for level_name, level_config in self.optimal_criteria['fibonacci_levels'].items():
                fib_price = swing_high - (price_range * level_config['ratio'])
                distance_pct = (current_price - fib_price) / fib_price * 100
                
                # Check if within entry window
                min_dist, max_dist = self.entry_windows[level_name]
                if min_dist <= distance_pct <= max_dist:
                    confluence = self._calculate_confluence(period_data, fib_price, date)
                    
                    day_opportunities[level_name] = {
                        'price': fib_price,
                        'distance_pct': distance_pct,
                        'base_multiplier': level_config['base_multiplier'],
                        'confluence_score': confluence,
                        'in_window': True
                    }
            
            opportunities.append(day_opportunities)
        
        backtest_data['Opportunities'] = opportunities
        self.backtest_data = backtest_data
        
        print(f"✅ Opportunities calculated for {len(self.backtest_data)} days")
    
    def _calculate_confluence(self, period_data: pd.DataFrame, fib_price: float, date: pd.Timestamp) -> float:
        """Calculate confluence score"""
        score = 1.0
        
        # Historical support/resistance
        tolerance = fib_price * 0.03
        touches = period_data[
            (period_data['Low'] <= fib_price + tolerance) &
            (period_data['High'] >= fib_price - tolerance)
        ]
        if len(touches) > 1:
            score += len(touches) * 0.1
        
        # Round numbers
        if fib_price % 2500 < 125 or fib_price % 2500 > 2375:
            score += 0.15
        elif fib_price % 1000 < 100 or fib_price % 1000 > 900:
            score += 0.1
        
        # Volume
        if date in self.btc_data.index:
            current_vol = self.btc_data.loc[date, 'Volume']
            avg_vol = period_data['Volume'].mean()
            if current_vol > avg_vol * 1.1:
                score += 0.1
        
        return min(score, 2.0)
    
    def _execute_strategy(self):
        """Execute the strategy"""
        print(f"\n💰 EXECUTING GOLDILOCKS PLUS STRATEGY")
        
        last_trade_time = None
        total_btc = 0
        
        for date, row in self.backtest_data.iterrows():
            current_price = row['Close']
            drawdown = row['Drawdown']
            fear_greed = row['Fear_Greed']
            days_since_ath = row['Days_Since_ATH']
            opportunities = row['Opportunities']
            volatility = row['Volatility']
            
            # Dynamic cooldown
            if last_trade_time:
                base_cooldown = self.optimal_criteria['cooldown_hours']
                
                if pd.notna(volatility):
                    if volatility > 70:
                        cooldown_hours = base_cooldown * 0.5  # 24 hours
                    elif volatility < 30:
                        cooldown_hours = base_cooldown * 1.5  # 72 hours
                    else:
                        cooldown_hours = base_cooldown
                else:
                    cooldown_hours = base_cooldown
                
                hours_since = (date - last_trade_time).total_seconds() / 3600
                if hours_since < cooldown_hours:
                    continue
            
            # Check entry criteria
            if (drawdown >= self.optimal_criteria['min_drawdown'] and
                fear_greed <= self.optimal_criteria['max_fear_greed'] and
                days_since_ath >= self.optimal_criteria['min_days_since_ath']):
                
                # Find best opportunity
                best_opportunity = None
                best_score = 0
                
                for level_name, opp in opportunities.items():
                    if opp.get('in_window', False):
                        leverage = self._calculate_leverage(drawdown, fear_greed, level_name)
                        
                        opportunity_score = (
                            opp['base_multiplier'] * 
                            leverage * 
                            opp['confluence_score'] * 
                            abs(opp['distance_pct']) * 
                            (1 + abs(drawdown) / 50)
                        )
                        
                        if opportunity_score > best_score:
                            best_score = opportunity_score
                            best_opportunity = {
                                'level': level_name,
                                'target_price': opp['price'],
                                'distance_pct': opp['distance_pct'],
                                'base_multiplier': opp['base_multiplier'],
                                'leverage': leverage,
                                'confluence': opp['confluence_score'],
                                'opportunity_score': opportunity_score
                            }
                
                # Execute trade
                if best_opportunity and best_score > 8:  # Lower threshold for more trades
                    remaining_capital = self.total_capital - self.capital_deployed
                    if remaining_capital > 500:  # Lower minimum
                        
                        # More aggressive position sizing
                        base_position = min(remaining_capital * 0.2, 12000)  # 20% of remaining, max $12K
                        leverage = best_opportunity['leverage']
                        total_position = base_position * leverage
                        
                        btc_acquired = total_position / current_price
                        
                        trade = {
                            'date': date,
                            'level': best_opportunity['level'],
                            'price': current_price,
                            'capital_invested': base_position,
                            'leverage': leverage,
                            'total_position': total_position,
                            'btc_acquired': btc_acquired,
                            'drawdown': drawdown,
                            'fear_greed': fear_greed,
                            'confluence': best_opportunity['confluence'],
                            'opportunity_score': best_opportunity['opportunity_score'],
                            'distance_pct': best_opportunity['distance_pct']
                        }
                        
                        self.trades.append(trade)
                        self.capital_deployed += base_position
                        total_btc += btc_acquired
                        last_trade_time = date
                        
                        print(f"   🎯 Trade {len(self.trades)}: {date.strftime('%Y-%m-%d')}")
                        print(f"      Level: {best_opportunity['level']} @ ${current_price:,.0f}")
                        print(f"      Drawdown: {drawdown:.1f}%, F&G: {fear_greed:.0f}")
                        print(f"      Leverage: {leverage:.1f}x, Position: ${total_position:,.0f}")
                        print(f"      BTC: {btc_acquired:.6f}")
        
        print(f"\n✅ Strategy executed: {len(self.trades)} trades")
        print(f"   Capital deployed: ${self.capital_deployed:,.0f}")
        print(f"   Total BTC: {total_btc:.6f}")
        
        # Check trade frequency
        years = (self.end_date - self.start_date).days / 365.25
        trades_per_year = len(self.trades) / years
        print(f"   Trades per year: {trades_per_year:.1f} (target: 15-20)")
        
        if 15 <= trades_per_year <= 20:
            print("   ✅ PERFECT! Hit the target trade frequency!")
        elif 10 <= trades_per_year < 15:
            print("   📊 Good frequency, close to target")
        elif trades_per_year > 20:
            print("   ⚠️ Slightly over-trading")
        else:
            print("   ⚠️ Still under-trading")
    
    def _calculate_leverage(self, drawdown: float, fear_greed: float, level_name: str) -> float:
        """Calculate optimal leverage"""
        base_leverage = self.optimal_criteria['leverage_scaling']['base_leverage']
        
        # Drawdown bonus
        drawdown_bonus = abs(drawdown) * self.optimal_criteria['leverage_scaling']['drawdown_multiplier'] / 10
        
        # Fear bonus
        fear_bonus = (35 - fear_greed) * self.optimal_criteria['leverage_scaling']['fear_multiplier'] / 10
        
        # Level multipliers
        level_multipliers = {
            '23.6%': 0.8,  # Lower for shallow retracement
            '38.2%': 1.0,
            '50.0%': 1.2,
            '61.8%': 1.5,
            '78.6%': 1.8
        }
        level_multiplier = level_multipliers.get(level_name, 1.0)
        
        total_leverage = (base_leverage + drawdown_bonus + fear_bonus) * level_multiplier
        
        return min(total_leverage, self.optimal_criteria['leverage_scaling']['max_leverage'])
    
    def _calculate_performance(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        # Performance calculation
        final_price = self.btc_data[self.btc_data.index <= self.end_date]['Close'].iloc[-1]
        total_btc = sum(trade['btc_acquired'] for trade in self.trades)
        final_value = total_btc * final_price
        
        total_return = (final_value - self.capital_deployed) / self.capital_deployed
        years = (self.end_date - self.start_date).days / 365.25
        annual_return = (final_value / self.capital_deployed) ** (1 / years) - 1
        
        # Buy & Hold comparison
        start_price = self.btc_data[self.btc_data.index >= self.start_date]['Close'].iloc[0]
        buy_hold_btc = self.total_capital / start_price
        buy_hold_final = buy_hold_btc * final_price
        buy_hold_return = (buy_hold_final - self.total_capital) / self.total_capital
        buy_hold_annual = (buy_hold_final / self.total_capital) ** (1 / years) - 1
        
        # Enhanced metrics
        trades_per_year = len(self.trades) / years
        avg_leverage = np.mean([trade['leverage'] for trade in self.trades])
        level_distribution = {}
        for trade in self.trades:
            level = trade['level']
            level_distribution[level] = level_distribution.get(level, 0) + 1
        
        return {
            'trades_executed': len(self.trades),
            'trades_per_year': trades_per_year,
            'capital_deployed': self.capital_deployed,
            'capital_efficiency': self.capital_deployed / self.total_capital,
            'total_btc': total_btc,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'buy_hold_final': buy_hold_final,
            'buy_hold_annual': buy_hold_annual,
            'outperformance': annual_return - buy_hold_annual,
            'avg_leverage': avg_leverage,
            'level_distribution': level_distribution,
            'avg_entry_price': self.capital_deployed / total_btc if total_btc > 0 else 0,
            'start_price': start_price,
            'final_price': final_price
        }
    
    def _display_results(self, results):
        """Display results"""
        if 'error' in results:
            print(f"❌ {results['error']}")
            return
        
        print(f"\n🎯 GOLDILOCKS PLUS RESULTS")
        print("="*60)
        
        print(f"📊 STRATEGY PERFORMANCE:")
        print(f"   Trades Executed: {results['trades_executed']}")
        print(f"   Trades per Year: {results['trades_per_year']:.1f} (target: 15-20)")
        print(f"   Capital Deployed: ${results['capital_deployed']:,.0f} ({results['capital_efficiency']:.1%})")
        print(f"   Total BTC: {results['total_btc']:.6f}")
        print(f"   Average Entry: ${results['avg_entry_price']:,.0f}")
        print(f"   Average Leverage: {results['avg_leverage']:.1f}x")
        print(f"   Final Value: ${results['final_value']:,.0f}")
        print(f"   Total Return: {results['total_return']:+.1%}")
        print(f"   Annual Return: {results['annual_return']:+.1%}")
        
        print(f"\n📊 FIBONACCI LEVEL DISTRIBUTION:")
        for level, count in results['level_distribution'].items():
            pct = count / results['trades_executed'] * 100
            print(f"   {level}: {count} trades ({pct:.1f}%)")
        
        print(f"\n📊 VS BUY & HOLD:")
        print(f"   Buy & Hold: ${results['buy_hold_final']:,.0f} ({results['buy_hold_annual']:+.1%})")
        print(f"   Outperformance: {results['outperformance']:+.1%}")
        
        # Target comparison
        target_annual = 2.454
        vs_target = results['annual_return'] / target_annual
        
        print(f"\n🎯 VS TARGET (+245.4%):")
        print(f"   Goldilocks Plus: {results['annual_return']:+.1%}")
        print(f"   vs Target: {vs_target:.1%}")
        
        # Success metrics
        optimal_trades = 15 <= results['trades_per_year'] <= 20
        beats_buy_hold = results['annual_return'] > results['buy_hold_annual']
        near_target = vs_target >= 0.8
        
        success_count = sum([optimal_trades, beats_buy_hold, near_target])
        
        if success_count >= 2:
            if results['annual_return'] > target_annual:
                grade = "A+ 🎉"
                recommendation = "TARGET ACHIEVED!"
            else:
                grade = "A 🎯"
                recommendation = "EXCELLENT BALANCE ACHIEVED!"
        elif beats_buy_hold:
            grade = "B+ 📊"
            recommendation = "Good balance, beats Buy & Hold"
        else:
            grade = "B 📈"
            recommendation = "Needs further tuning"
        
        print(f"\n🏆 GOLDILOCKS PLUS GRADE: {grade}")
        print(f"💡 RECOMMENDATION: {recommendation}")
        
        if optimal_trades:
            print(f"\n🎊 TRADE FREQUENCY SUCCESS!")
            print(f"✅ Achieved optimal 15-20 trades per year balance")

async def main():
    """Run Goldilocks Plus analysis"""
    print("🎯 GOLDILOCKS PLUS NANPIN - Aggressive Balance")
    print("=" * 60)
    
    periods = [
        {"name": "Full Period (G+)", "start": "2020-01-01", "end": "2024-12-31"},
        {"name": "COVID Era (G+)", "start": "2020-01-01", "end": "2021-12-31"},
        {"name": "Bear Market (G+)", "start": "2021-11-01", "end": "2022-12-31"},
        {"name": "Recovery (G+)", "start": "2023-01-01", "end": "2024-12-31"}
    ]
    
    target_achieved = False
    best_performance = 0
    
    for period in periods:
        print(f"\n🔄 Testing {period['name']} ({period['start']} to {period['end']})...")
        
        strategy = GoldilocksPlusNanpin(period['start'], period['end'])
        results = await strategy.run_goldilocks_plus_backtest()
        
        if results and 'annual_return' in results:
            if results['annual_return'] > 2.454:
                target_achieved = True
            if results['annual_return'] > best_performance:
                best_performance = results['annual_return']
    
    print(f"\n🏆 GOLDILOCKS PLUS SUMMARY:")
    print(f"   Best Performance: {best_performance:+.1%}")
    print(f"   vs Target: {best_performance/2.454:.1%}")
    
    if target_achieved:
        print(f"\n🎊 🎯 TARGET ACHIEVED! 🎯 🎊")
        print(f"🌸 Goldilocks Plus found the perfect balance!")

if __name__ == "__main__":
    asyncio.run(main())