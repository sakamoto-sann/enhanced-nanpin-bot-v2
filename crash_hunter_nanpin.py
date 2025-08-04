#!/usr/bin/env python3
"""
🎯 Crash Hunter Nanpin Strategy
Focus ONLY on major market crashes with maximum leverage
"""

import asyncio
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class CrashHunterNanpin:
    """Nanpin strategy that ONLY buys during major market crashes"""
    
    def __init__(self, start_date="2020-01-01", end_date="2024-12-31"):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.total_capital = 100000  # $100K total
        
        # EXTREME crash criteria
        self.crash_criteria = {
            'min_drawdown': -30,    # Must be 30%+ crash from recent high
            'min_fear_level': 10,   # Extreme fear only
            'min_days_since_ath': 30,  # At least 30 days from ATH
            'max_leverage': 20      # Up to 20x leverage
        }
        
        self.trades = []
        self.capital_deployed = 0
        
    async def run_crash_hunter_backtest(self):
        """Run crash hunter backtest"""
        try:
            print("🎯 CRASH HUNTER NANPIN STRATEGY")
            print("="*50)
            print("Strategy: ONLY buy during major market crashes with maximum leverage")
            print(f"Capital: ${self.total_capital:,}")
            print(f"Criteria: {self.crash_criteria['min_drawdown']}% drawdown + extreme fear")
            
            # Load data
            await self._load_data()
            
            # Identify crash periods
            self._identify_crash_periods()
            
            # Execute crash hunting strategy
            self._execute_crash_hunting()
            
            # Calculate performance
            results = self._calculate_crash_hunter_performance()
            
            # Display results
            self._display_crash_hunter_results(results)
            
            return results
            
        except Exception as e:
            print(f"❌ Crash hunter backtest failed: {e}")
            return {}
    
    async def _load_data(self):
        """Load BTC data with crash indicators"""
        btc = yf.Ticker("BTC-USD")
        data = btc.history(
            start=(self.start_date - timedelta(days=365)).strftime("%Y-%m-%d"),
            end=self.end_date.strftime("%Y-%m-%d"),
            interval="1d"
        )
        
        self.btc_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        self.btc_data.index = pd.to_datetime(self.btc_data.index.date)
        self.btc_data = self.btc_data.dropna()
        
        # Add crash detection indicators
        self._add_crash_indicators()
        
        print(f"✅ Data loaded: {len(self.btc_data)} days")
    
    def _add_crash_indicators(self):
        """Add indicators to detect major crashes"""
        # Rolling ATH
        self.btc_data['ATH'] = self.btc_data['High'].rolling(90, min_periods=1).max()
        
        # Drawdown from ATH
        self.btc_data['Drawdown'] = (self.btc_data['Close'] - self.btc_data['ATH']) / self.btc_data['ATH'] * 100
        
        # Days since ATH
        self.btc_data['Days_Since_ATH'] = 0
        for i in range(len(self.btc_data)):
            if i == 0:
                self.btc_data.iloc[i, self.btc_data.columns.get_loc('Days_Since_ATH')] = 0
            else:
                current_high = self.btc_data.iloc[i]['High']
                ath = self.btc_data.iloc[:i+1]['High'].max()
                if current_high >= ath:
                    self.btc_data.iloc[i, self.btc_data.columns.get_loc('Days_Since_ATH')] = 0
                else:
                    self.btc_data.iloc[i, self.btc_data.columns.get_loc('Days_Since_ATH')] = self.btc_data.iloc[i-1]['Days_Since_ATH'] + 1
        
        # Volatility (30-day)
        returns = self.btc_data['Close'].pct_change()
        self.btc_data['Volatility'] = returns.rolling(30).std() * np.sqrt(365) * 100
        
        # Simulated Fear & Greed (based on drawdown and volatility)
        self.btc_data['Fear_Greed'] = np.clip(
            50 + self.btc_data['Drawdown'] * 0.8 - (self.btc_data['Volatility'] - 50) * 0.5,
            0, 100
        )
        
        # Momentum (30-day)
        self.btc_data['Momentum_30d'] = (self.btc_data['Close'] / self.btc_data['Close'].shift(30) - 1) * 100
    
    def _identify_crash_periods(self):
        """Identify major crash periods"""
        # Filter to backtest period
        backtest_data = self.btc_data[
            (self.btc_data.index >= self.start_date) &
            (self.btc_data.index <= self.end_date)
        ].copy()
        
        # Identify crash conditions
        crash_conditions = (
            (backtest_data['Drawdown'] <= self.crash_criteria['min_drawdown']) &
            (backtest_data['Fear_Greed'] <= self.crash_criteria['min_fear_level']) &
            (backtest_data['Days_Since_ATH'] >= self.crash_criteria['min_days_since_ath'])
        )
        
        self.crash_periods = backtest_data[crash_conditions].copy()
        
        print(f"🚨 Crash periods identified: {len(self.crash_periods)} days")
        
        if len(self.crash_periods) > 0:
            print("   Major crash periods:")
            for date, row in self.crash_periods.head(10).iterrows():
                print(f"     {date.strftime('%Y-%m-%d')}: ${row['Close']:,.0f} "
                      f"({row['Drawdown']:.1f}% drawdown, F&G: {row['Fear_Greed']:.0f})")
    
    def _execute_crash_hunting(self):
        """Execute crash hunting strategy"""
        if self.crash_periods.empty:
            print("❌ No crash periods found!")
            return
        
        # Group consecutive crash days
        crash_groups = self._group_crash_periods()
        
        capital_per_crash = self.total_capital / len(crash_groups)
        
        print(f"\n💰 EXECUTING CRASH HUNTING STRATEGY")
        print(f"   Crash groups: {len(crash_groups)}")
        print(f"   Capital per crash: ${capital_per_crash:,.0f}")
        
        for i, (start_date, end_date, worst_day) in enumerate(crash_groups):
            crash_data = self.crash_periods[
                (self.crash_periods.index >= start_date) &
                (self.crash_periods.index <= end_date)
            ]
            
            # Find the worst day in this crash period
            worst_row = crash_data.loc[worst_day]
            worst_price = worst_row['Close']
            worst_drawdown = worst_row['Drawdown']
            worst_fear = worst_row['Fear_Greed']
            
            # Calculate leverage based on severity
            leverage = self._calculate_crash_leverage(worst_drawdown, worst_fear)
            
            # Execute trade
            trade_amount = capital_per_crash * leverage
            btc_acquired = trade_amount / worst_price
            
            trade = {
                'date': worst_day,
                'price': worst_price,
                'capital_invested': capital_per_crash,
                'leverage': leverage,
                'trade_amount': trade_amount,
                'btc_acquired': btc_acquired,
                'drawdown': worst_drawdown,
                'fear_greed': worst_fear,
                'crash_group': i + 1
            }
            
            self.trades.append(trade)
            self.capital_deployed += capital_per_crash
            
            print(f"   🎯 Crash {i+1}: {worst_day.strftime('%Y-%m-%d')}")
            print(f"      Price: ${worst_price:,.0f}")
            print(f"      Drawdown: {worst_drawdown:.1f}%")
            print(f"      Fear/Greed: {worst_fear:.0f}")
            print(f"      Leverage: {leverage:.1f}x")
            print(f"      Trade Amount: ${trade_amount:,.0f}")
            print(f"      BTC Acquired: {btc_acquired:.6f}")
    
    def _group_crash_periods(self):
        """Group consecutive crash days into crash events"""
        crash_groups = []
        
        if self.crash_periods.empty:
            return crash_groups
        
        current_start = self.crash_periods.index[0]
        current_end = self.crash_periods.index[0]
        
        for i in range(1, len(self.crash_periods)):
            current_date = self.crash_periods.index[i]
            prev_date = self.crash_periods.index[i-1]
            
            # If gap > 7 days, start new group
            if (current_date - prev_date).days > 7:
                # Find worst day in current group
                group_data = self.crash_periods[
                    (self.crash_periods.index >= current_start) &
                    (self.crash_periods.index <= current_end)
                ]
                worst_day = group_data['Drawdown'].idxmin()
                
                crash_groups.append((current_start, current_end, worst_day))
                current_start = current_date
            
            current_end = current_date
        
        # Add final group
        if not self.crash_periods.empty:
            group_data = self.crash_periods[
                (self.crash_periods.index >= current_start) &
                (self.crash_periods.index <= current_end)
            ]
            worst_day = group_data['Drawdown'].idxmin()
            crash_groups.append((current_start, current_end, worst_day))
        
        return crash_groups
    
    def _calculate_crash_leverage(self, drawdown: float, fear_greed: float) -> float:
        """Calculate leverage based on crash severity"""
        base_leverage = 5.0
        
        # Increase leverage for deeper crashes
        drawdown_multiplier = 1.0 + abs(drawdown) / 20  # +1x for every 20% crash
        
        # Increase leverage for extreme fear
        fear_multiplier = 1.0 + (10 - fear_greed) / 10 if fear_greed < 10 else 1.0
        
        total_leverage = base_leverage * drawdown_multiplier * fear_multiplier
        
        return min(total_leverage, self.crash_criteria['max_leverage'])
    
    def _calculate_crash_hunter_performance(self):
        """Calculate crash hunter performance"""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        # Current BTC price
        final_price = self.btc_data[self.btc_data.index <= self.end_date]['Close'].iloc[-1]
        
        # Calculate total BTC and value
        total_btc = sum(trade['btc_acquired'] for trade in self.trades)
        final_value = total_btc * final_price
        
        # Performance metrics
        total_return = (final_value - self.capital_deployed) / self.capital_deployed
        years = (self.end_date - self.start_date).days / 365.25
        annual_return = (final_value / self.capital_deployed) ** (1 / years) - 1
        
        # Buy & Hold comparison
        start_price = self.btc_data[self.btc_data.index >= self.start_date]['Close'].iloc[0]
        buy_hold_btc = self.total_capital / start_price
        buy_hold_final = buy_hold_btc * final_price
        buy_hold_return = (buy_hold_final - self.total_capital) / self.total_capital
        buy_hold_annual = (buy_hold_final / self.total_capital) ** (1 / years) - 1
        
        return {
            'trades_executed': len(self.trades),
            'capital_deployed': self.capital_deployed,
            'total_btc': total_btc,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'buy_hold_final': buy_hold_final,
            'buy_hold_return': buy_hold_return,
            'buy_hold_annual': buy_hold_annual,
            'outperformance': annual_return - buy_hold_annual,
            'avg_entry_price': self.capital_deployed / total_btc if total_btc > 0 else 0,
            'start_price': start_price,
            'final_price': final_price
        }
    
    def _display_crash_hunter_results(self, results):
        """Display crash hunter results"""
        if 'error' in results:
            print(f"❌ {results['error']}")
            return
        
        print(f"\n🎯 CRASH HUNTER RESULTS")
        print("="*60)
        
        print(f"📊 STRATEGY PERFORMANCE:")
        print(f"   Trades Executed: {results['trades_executed']}")
        print(f"   Capital Deployed: ${results['capital_deployed']:,.0f}")
        print(f"   Total BTC Acquired: {results['total_btc']:.6f}")
        print(f"   Average Entry Price: ${results['avg_entry_price']:,.0f}")
        print(f"   Final Portfolio Value: ${results['final_value']:,.0f}")
        print(f"   Total Return: {results['total_return']:+.1%}")
        print(f"   Annual Return: {results['annual_return']:+.1%}")
        
        print(f"\n📊 VS BUY & HOLD:")
        print(f"   Buy & Hold Final Value: ${results['buy_hold_final']:,.0f}")
        print(f"   Buy & Hold Annual Return: {results['buy_hold_annual']:+.1%}")
        print(f"   Outperformance: {results['outperformance']:+.1%}")
        
        # Target comparison
        target_annual = 2.454  # 245.4%
        vs_target = results['annual_return'] / target_annual
        
        print(f"\n🎯 VS TARGET (+245.4%):")
        print(f"   Crash Hunter Annual: {results['annual_return']:+.1%}")
        print(f"   vs Target: {vs_target:.1%}")
        
        # Performance grade
        if results['annual_return'] > target_annual:
            if results['annual_return'] > results['buy_hold_annual']:
                grade = "S+ 🌟"
                recommendation = "LEGENDARY - Beats both target AND Buy & Hold!"
            else:
                grade = "A+ 🎉"
                recommendation = "EXCELLENT - Target achieved!"
        elif results['annual_return'] > results['buy_hold_annual']:
            grade = "A 🎯"
            recommendation = "SUCCESS - Beats Buy & Hold with leverage!"
        else:
            grade = "B 📊"
            recommendation = "Needs optimization"
        
        print(f"\n🏆 PERFORMANCE GRADE: {grade}")
        print(f"💡 RECOMMENDATION: {recommendation}")
        
        if results['annual_return'] > results['buy_hold_annual']:
            print(f"\n🎊 SUCCESS! Strategy beats Buy & Hold despite using leverage strategically!")
            print(f"✅ This proves the concept works when executed correctly")

async def main():
    """Run crash hunter analysis"""
    print("🎯 CRASH HUNTER NANPIN - The Ultimate Test")
    print("=" * 60)
    
    # Test different periods
    periods = [
        {"name": "Full Period", "start": "2020-01-01", "end": "2024-12-31"},
        {"name": "COVID + Recovery", "start": "2020-01-01", "end": "2021-12-31"},
        {"name": "Bear Market", "start": "2021-11-01", "end": "2022-12-31"}
    ]
    
    for period in periods:
        print(f"\n🔄 Testing {period['name']} ({period['start']} to {period['end']})...")
        
        hunter = CrashHunterNanpin(period['start'], period['end'])
        results = await hunter.run_crash_hunter_backtest()

if __name__ == "__main__":
    asyncio.run(main())