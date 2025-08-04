#!/usr/bin/env python3
"""
🎯 QUICK CONFIDENCE ANALYSIS
Analyzes current confidence patterns and provides threshold recommendations
"""

import asyncio
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

load_dotenv(override=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exchanges.backpack_client_fixed import BackpackNanpinClient

class QuickConfidenceAnalyzer:
    """
    🎯 Quick Confidence Analysis Based on Current Patterns
    """
    
    def __init__(self):
        """Initialize analyzer"""
        self.api_key = os.getenv('BACKPACK_API_KEY')
        self.secret_key = os.getenv('BACKPACK_SECRET_KEY')
        self.backpack_client = None
        
    async def initialize(self):
        """Initialize Backpack client"""
        self.backpack_client = BackpackNanpinClient(self.api_key, self.secret_key)
        await self.backpack_client._init_session()
        
    async def get_recent_price_data(self, days: int = 120):
        """Get recent price data for analysis"""
        try:
            print(f"📊 Fetching {days} days of BTC futures price data...")
            
            # Get daily klines for past 4 months
            klines = await self.backpack_client.get_klines(
                symbol='BTC_USDC_PERP',
                interval='1d', 
                limit=days
            )
            
            if not klines:
                print("❌ No historical data available")
                return None
                
            # Convert to DataFrame
            price_data = []
            for candle in klines:
                if isinstance(candle, dict) and 'start' in candle:
                    price_data.append({
                        'date': pd.to_datetime(candle['start']),
                        'open': float(candle['open']),
                        'high': float(candle['high']),
                        'low': float(candle['low']),
                        'close': float(candle['close']),
                        'volume': float(candle['volume'])
                    })
            
            df = pd.DataFrame(price_data)
            df.set_index('date', inplace=True)
            df = df.sort_index()
            
            print(f"✅ Retrieved {len(df)} days of data")
            print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            print(f"   Price range: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Failed to get price data: {e}")
            return None
    
    def calculate_market_conditions(self, df):
        """Calculate various market condition indicators"""
        # Calculate returns
        df['daily_return'] = df['close'].pct_change()
        df['volatility_7d'] = df['daily_return'].rolling(7).std() * np.sqrt(7) * 100
        df['volatility_30d'] = df['daily_return'].rolling(30).std() * np.sqrt(30) * 100
        
        # Calculate drawdown from rolling high
        df['rolling_max'] = df['close'].expanding().max()
        df['drawdown'] = (df['close'] - df['rolling_max']) / df['rolling_max'] * 100
        
        # Calculate RSI-like momentum
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate moving averages
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_50'] = df['close'].rolling(50).mean()
        
        # Price position relative to MAs
        df['price_vs_ma20'] = (df['close'] - df['ma_20']) / df['ma_20'] * 100
        df['price_vs_ma50'] = (df['close'] - df['ma_50']) / df['ma_50'] * 100
        
        return df
    
    def simulate_confidence_scores(self, df):
        """Simulate confidence scores based on market conditions"""
        confidence_scores = []
        
        for i, (date, row) in enumerate(df.iterrows()):
            if i < 50:  # Need enough data for indicators
                confidence_scores.append(np.nan)
                continue
                
            # Base confidence from various factors
            base_confidence = 0.3  # 30% base
            
            # Volatility factor (higher volatility = lower confidence initially, but opportunities)
            vol_factor = 0
            if pd.notna(row['volatility_30d']):
                if 20 < row['volatility_30d'] < 80:  # Sweet spot for nanpin
                    vol_factor = 0.15
                elif row['volatility_30d'] > 100:  # High vol = more opportunities but riskier
                    vol_factor = 0.10
                else:
                    vol_factor = 0.05
            
            # Drawdown factor (larger drawdowns = more opportunities)
            dd_factor = 0
            if pd.notna(row['drawdown']):
                if row['drawdown'] < -15:  # Significant drawdown
                    dd_factor = 0.20
                elif row['drawdown'] < -10:
                    dd_factor = 0.15
                elif row['drawdown'] < -5:
                    dd_factor = 0.10
                else:
                    dd_factor = 0.05
            
            # RSI factor (oversold conditions)
            rsi_factor = 0
            if pd.notna(row['rsi']):
                if row['rsi'] < 30:  # Oversold
                    rsi_factor = 0.15
                elif row['rsi'] < 40:
                    rsi_factor = 0.10
                else:
                    rsi_factor = 0.05
            
            # Trend factor (position relative to MAs)
            trend_factor = 0
            if pd.notna(row['price_vs_ma20']) and pd.notna(row['price_vs_ma50']):
                if row['price_vs_ma20'] < -5 and row['price_vs_ma50'] < -10:  # Below MAs = opportunity
                    trend_factor = 0.15
                elif row['price_vs_ma20'] < 0:
                    trend_factor = 0.10
                else:
                    trend_factor = 0.05
            
            # Calculate total confidence
            total_confidence = base_confidence + vol_factor + dd_factor + rsi_factor + trend_factor
            
            # Add some randomness to simulate real conditions
            noise = np.random.normal(0, 0.05)  # 5% standard deviation
            total_confidence += noise
            
            # Clamp between 0 and 1
            total_confidence = max(0.1, min(0.9, total_confidence))
            
            confidence_scores.append(total_confidence)
        
        df['confidence_score'] = confidence_scores
        return df
    
    def analyze_optimal_threshold(self, df):
        """Analyze optimal confidence threshold"""
        # Remove NaN values
        df_clean = df.dropna(subset=['confidence_score'])
        
        if len(df_clean) == 0:
            print("❌ No valid confidence data")
            return
        
        print("\n" + "="*80)
        print("🎯 NANPIN BOT CONFIDENCE THRESHOLD ANALYSIS")
        print("="*80)
        
        # Basic statistics
        conf_stats = {
            'count': len(df_clean),
            'mean': df_clean['confidence_score'].mean(),
            'median': df_clean['confidence_score'].median(),
            'std': df_clean['confidence_score'].std(),
            'min': df_clean['confidence_score'].min(),
            'max': df_clean['confidence_score'].max()
        }
        
        print(f"\n📊 CONFIDENCE STATISTICS (Past 4 Months)")
        print(f"   Total Days: {conf_stats['count']}")
        print(f"   Average: {conf_stats['mean']:.1%}")
        print(f"   Median: {conf_stats['median']:.1%}")
        print(f"   Std Dev: {conf_stats['std']:.1%}")
        print(f"   Range: {conf_stats['min']:.1%} - {conf_stats['max']:.1%}")
        
        # Percentile analysis
        percentiles = [10, 25, 50, 75, 90, 95]
        print(f"\n📈 CONFIDENCE PERCENTILES")
        for p in percentiles:
            value = df_clean['confidence_score'].quantile(p/100)
            print(f"   {p:2d}th percentile: {value:.1%}")
        
        # Threshold analysis
        test_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        
        print(f"\n🎯 THRESHOLD ANALYSIS")
        print("   Threshold | Days | % Time | Trades/Month | Monthly Opp.")
        print("   " + "-"*60)
        
        best_threshold = None
        best_score = 0
        
        for threshold in test_thresholds:
            qualifying_days = df_clean[df_clean['confidence_score'] >= threshold]
            pct_time = len(qualifying_days) / len(df_clean) * 100
            trades_per_month = len(qualifying_days) / 4  # 4 months of data
            
            # Score based on reasonable trading frequency (2-8 trades/month)
            if 2 <= trades_per_month <= 8:
                # Prefer higher threshold with reasonable frequency
                score = trades_per_month * (threshold ** 2)
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            print(f"   {threshold:7.0%}   | {len(qualifying_days):4d} | {pct_time:6.1f}% | {trades_per_month:11.1f} | {len(qualifying_days):10d}")
        
        # Monthly analysis
        df_clean['month'] = df_clean.index.to_period('M')
        monthly_stats = df_clean.groupby('month').agg({
            'confidence_score': ['mean', 'max', 'count'],
            'close': ['first', 'last'],
            'daily_return': 'sum'
        }).round(3)
        
        print(f"\n📅 MONTHLY ANALYSIS")
        print("   Month    | Avg Conf. | Max Conf. | Days | BTC Return")
        print("   " + "-"*55)
        
        total_return = 0
        high_conf_months = 0
        
        for month in monthly_stats.index:
            avg_conf = monthly_stats.loc[month, ('confidence_score', 'mean')]
            max_conf = monthly_stats.loc[month, ('confidence_score', 'max')]
            days = monthly_stats.loc[month, ('confidence_score', 'count')]
            monthly_return = monthly_stats.loc[month, ('daily_return', 'sum')] * 100
            
            total_return += monthly_return
            if avg_conf > 0.5:
                high_conf_months += 1
            
            print(f"   {str(month)} | {avg_conf:8.1%} | {max_conf:8.1%} | {days:4.0f} | {monthly_return:8.1f}%")
        
        print(f"\n💡 ANALYSIS RESULTS:")
        
        if best_threshold:
            optimal_trades = len(df_clean[df_clean['confidence_score'] >= best_threshold]) / 4
            print(f"   🎯 Recommended Threshold: {best_threshold:.0%}")
            print(f"   📊 Expected Trading Frequency: {optimal_trades:.1f} trades/month")
            print(f"   📈 Trading Time: {len(df_clean[df_clean['confidence_score'] >= best_threshold])/len(df_clean)*100:.1f}% of days")
        else:
            print(f"   ⚠️  No optimal threshold in 2-8 trades/month range found")
            
        # Current bot analysis (49.1% vs thresholds)
        current_conf = 0.491
        print(f"\n🤖 CURRENT BOT ANALYSIS:")
        print(f"   Current Confidence: {current_conf:.1%}")
        print(f"   Current Threshold: 65%")
        
        if current_conf < 0.65:
            days_above_current = len(df_clean[df_clean['confidence_score'] >= current_conf])
            days_above_threshold = len(df_clean[df_clean['confidence_score'] >= 0.65])
            
            print(f"   Days ≥ {current_conf:.1%}: {days_above_current} ({days_above_current/len(df_clean)*100:.1f}%)")
            print(f"   Days ≥ 65%: {days_above_threshold} ({days_above_threshold/len(df_clean)*100:.1f}%)")
            
            if days_above_threshold < len(df_clean) * 0.1:  # Less than 10% of time
                print(f"   ⚠️  65% threshold may be too high - only {days_above_threshold/len(df_clean)*100:.1f}% of days qualify")
                
                # Suggest alternative
                alt_threshold = df_clean['confidence_score'].quantile(0.8)  # 80th percentile
                alt_days = len(df_clean[df_clean['confidence_score'] >= alt_threshold])
                print(f"   💡 Consider {alt_threshold:.0%} threshold: {alt_days} days ({alt_days/len(df_clean)*100:.1f}%), {alt_days/4:.1f} trades/month")
        
        print(f"\n🏆 FINAL RECOMMENDATIONS:")
        
        # Based on current confidence vs historical patterns
        historical_median = conf_stats['median']
        if current_conf < historical_median:
            print(f"   1. Current confidence ({current_conf:.1%}) is below historical median ({historical_median:.1%})")
            print(f"   2. Consider temporary threshold of {historical_median:.0%} until conditions improve")
        else:
            print(f"   1. Current confidence ({current_conf:.1%}) is reasonable")
        
        if best_threshold:
            print(f"   3. Long-term optimal threshold: {best_threshold:.0%}")
            print(f"   4. Expected {optimal_trades:.1f} trades per month at optimal threshold")
        
        # Conservative vs aggressive recommendations
        conservative_threshold = df_clean['confidence_score'].quantile(0.75)  # 75th percentile
        aggressive_threshold = df_clean['confidence_score'].quantile(0.6)     # 60th percentile
        
        cons_trades = len(df_clean[df_clean['confidence_score'] >= conservative_threshold]) / 4
        agg_trades = len(df_clean[df_clean['confidence_score'] >= aggressive_threshold]) / 4
        
        print(f"\n   📊 STRATEGY OPTIONS:")
        print(f"   Conservative ({conservative_threshold:.0%}): {cons_trades:.1f} trades/month")
        print(f"   Aggressive ({aggressive_threshold:.0%}): {agg_trades:.1f} trades/month")
        print(f"   Current (65%): ~{len(df_clean[df_clean['confidence_score'] >= 0.65])/4:.1f} trades/month")
        
        print("="*80)

async def main():
    """Run quick confidence analysis"""
    analyzer = QuickConfidenceAnalyzer()
    
    try:
        await analyzer.initialize()
        
        # Get historical data
        df = await analyzer.get_recent_price_data(120)  # 4 months
        if df is None:
            return
        
        # Calculate market conditions
        df = analyzer.calculate_market_conditions(df)
        
        # Simulate confidence scores based on conditions
        df = analyzer.simulate_confidence_scores(df)
        
        # Analyze optimal threshold
        analyzer.analyze_optimal_threshold(df)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if analyzer.backpack_client and hasattr(analyzer.backpack_client, 'session'):
            if analyzer.backpack_client.session:
                await analyzer.backpack_client.session.close()

if __name__ == "__main__":
    asyncio.run(main())