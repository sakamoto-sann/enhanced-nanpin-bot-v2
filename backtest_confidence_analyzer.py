#!/usr/bin/env python3
"""
🎯 NANPIN BOT CONFIDENCE THRESHOLD BACKTESTING
Analyzes past 4 months to determine optimal confidence thresholds for futures trading
"""

import asyncio
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
import logging

load_dotenv(override=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exchanges.backpack_client_fixed import BackpackNanpinClient
from core.macro_analyzer import MacroAnalyzer  
from data.enhanced_liquidation_aggregator import EnhancedLiquidationAggregator
from strategies.macro_enhanced_goldilocks_strategy import MacroEnhancedGoldilocksStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfidenceBacktester:
    """
    🎯 Confidence Threshold Backtesting System
    
    Analyzes historical data to determine optimal confidence thresholds
    """
    
    def __init__(self):
        """Initialize backtester"""
        self.api_key = os.getenv('BACKPACK_API_KEY')
        self.secret_key = os.getenv('BACKPACK_SECRET_KEY')
        
        # Initialize components
        self.backpack_client = None
        self.macro_analyzer = None
        self.liquidation_aggregator = None
        self.strategy = None
        
        # Backtest configuration
        self.symbol = 'BTC_USDC_PERP'
        self.start_date = datetime.now() - timedelta(days=120)  # Past 4 months
        self.end_date = datetime.now()
        
        # Performance tracking
        self.backtest_results = []
        self.confidence_analysis = {}
        
    async def initialize_components(self):
        """Initialize all trading components"""
        logger.info("🚀 Initializing backtesting components...")
        
        # Initialize Backpack client
        self.backpack_client = BackpackNanpinClient(self.api_key, self.secret_key)
        await self.backpack_client.initialize()
        
        # Initialize macro analyzer
        self.macro_analyzer = MacroAnalyzer()
        
        # Initialize liquidation aggregator
        self.liquidation_aggregator = EnhancedLiquidationAggregator(self.backpack_client)
        
        # Initialize strategy with test configuration
        strategy_config = {
            'min_drawdown': 15,
            'max_fear_greed': 80,
            'min_days_since_ath': 7,
            'base_leverage': 3.0,
            'max_leverage': 18.0,
            'cooldown_hours': 4,
            'dynamic_cooldown': True,
            'drawdown_multiplier': 1.5,
            'fear_multiplier': 1.2,
            'min_remaining_capital': 100,
            'base_position_pct': 10.0,
            'max_single_position': 15.0,
            'entry_windows': {
                '23.6%': {'confidence_boost': 1.1, 'enabled': True},
                '38.2%': {'confidence_boost': 1.2, 'enabled': True},
                '50.0%': {'confidence_boost': 1.15, 'enabled': True},
                '61.8%': {'confidence_boost': 1.25, 'enabled': True},
                '78.6%': {'confidence_boost': 1.3, 'enabled': True}
            }
        }
        
        self.strategy = MacroEnhancedGoldilocksStrategy(
            self.backpack_client,
            self.macro_analyzer,
            self.liquidation_aggregator,
            strategy_config
        )
        
        logger.info("✅ All components initialized")
    
    async def get_historical_data(self, days_back: int = 120):
        """Get historical price data from Backpack"""
        try:
            logger.info(f"📊 Fetching {days_back} days of historical data...")
            
            # Get daily klines for past 4 months
            klines = await self.backpack_client.get_klines(
                symbol=self.symbol,
                interval='1d',
                limit=days_back
            )
            
            if not klines:
                logger.error("❌ No historical data available")
                return None
            
            # Convert to DataFrame
            price_data = []
            for candle in klines:
                if isinstance(candle, dict) and 'start' in candle:
                    price_data.append({
                        'timestamp': pd.to_datetime(candle['start']),
                        'open': float(candle['open']),
                        'high': float(candle['high']),
                        'low': float(candle['low']),
                        'close': float(candle['close']),
                        'volume': float(candle['volume'])
                    })
            
            df = pd.DataFrame(price_data)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            logger.info(f"✅ Retrieved {len(df)} days of data")
            logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")
            logger.info(f"   Price range: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to get historical data: {e}")
            return None
    
    async def simulate_trading_day(self, date: datetime, price_data: pd.Series):
        """Simulate trading analysis for a specific day"""
        try:
            # Create market data for analysis
            enhanced_market_data = await self.liquidation_aggregator.get_enhanced_market_data('BTC')
            if not enhanced_market_data:
                return None
            
            # Override with historical price
            enhanced_market_data.current_price = price_data['close']
            
            # Create historical DataFrame for strategy analysis
            historical_df = pd.DataFrame({
                'open': [price_data['open']],
                'high': [price_data['high']],
                'low': [price_data['low']],
                'close': [price_data['close']],
                'volume': [price_data['volume']]
            }, index=[date])
            
            market_data = {
                'current_price': enhanced_market_data.current_price,
                'historical_data': historical_df,
                'enhanced_data': enhanced_market_data
            }
            
            # Analyze market conditions
            analysis = await self.strategy.analyze_market_conditions(market_data)
            
            if 'error' not in analysis:
                return {
                    'date': date,
                    'price': price_data['close'],
                    'confidence_score': analysis.get('confidence_score', 0),
                    'recommendations': analysis.get('recommendations', []),
                    'macro_regime': analysis.get('macro_analysis', {}).get('regime', 'unknown'),
                    'macro_signal': analysis.get('macro_analysis', {}).get('signal', 'unknown'),
                    'liquidation_clusters': len(analysis.get('liquidation_analysis', {}).get('clusters', [])),
                    'cascade_risk': analysis.get('liquidation_analysis', {}).get('cascade_risk', 0),
                    'market_stress': analysis.get('flipside_analysis', {}).get('signals', {}).get('market_stress', 50)
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"❌ Simulation error for {date}: {e}")
            return None
    
    async def run_historical_backtest(self):
        """Run complete historical backtest"""
        logger.info("🎯 Starting 4-month confidence threshold backtest...")
        
        # Get historical data
        historical_data = await self.get_historical_data(120)
        if historical_data is None:
            return
        
        # Simulate each trading day
        simulation_results = []
        total_days = len(historical_data)
        
        logger.info(f"📊 Running simulation for {total_days} trading days...")
        
        for i, (date, price_data) in enumerate(historical_data.iterrows()):
            if i % 10 == 0:
                logger.info(f"   Progress: {i+1}/{total_days} ({(i+1)/total_days*100:.1f}%)")
            
            result = await self.simulate_trading_day(date, price_data)
            if result:
                simulation_results.append(result)
            
            # Small delay to prevent API overload
            await asyncio.sleep(0.1)
        
        self.backtest_results = simulation_results
        logger.info(f"✅ Completed simulation: {len(simulation_results)} valid results")
        
        # Analyze results
        await self.analyze_confidence_patterns()
        
    def analyze_confidence_patterns(self):
        """Analyze confidence patterns and optimal thresholds"""
        logger.info("📊 Analyzing confidence patterns...")
        
        if not self.backtest_results:
            logger.error("❌ No backtest results to analyze")
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.backtest_results)
        
        # Basic statistics
        confidence_stats = {
            'total_days': len(df),
            'avg_confidence': df['confidence_score'].mean(),
            'median_confidence': df['confidence_score'].median(),
            'std_confidence': df['confidence_score'].std(),
            'min_confidence': df['confidence_score'].min(),
            'max_confidence': df['confidence_score'].max(),
            'percentiles': {
                '10th': df['confidence_score'].quantile(0.1),
                '25th': df['confidence_score'].quantile(0.25),
                '50th': df['confidence_score'].quantile(0.5),
                '75th': df['confidence_score'].quantile(0.75),
                '90th': df['confidence_score'].quantile(0.9),
                '95th': df['confidence_score'].quantile(0.95)
            }
        }
        
        # Count recommendations by confidence threshold
        threshold_analysis = {}
        test_thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        
        for threshold in test_thresholds:
            qualifying_days = df[df['confidence_score'] >= threshold]
            
            # Count BUY recommendations
            buy_recommendations = 0
            for _, row in qualifying_days.iterrows():
                for rec in row['recommendations']:
                    if rec.get('action') == 'BUY':
                        buy_recommendations += 1
                        break
            
            threshold_analysis[threshold] = {
                'qualifying_days': len(qualifying_days),
                'percentage_of_days': len(qualifying_days) / len(df) * 100,
                'buy_opportunities': buy_recommendations,
                'avg_confidence': qualifying_days['confidence_score'].mean() if len(qualifying_days) > 0 else 0,
                'trades_per_month': buy_recommendations / 4 if buy_recommendations > 0 else 0
            }
        
        # Market regime analysis
        regime_analysis = df.groupby('macro_regime').agg({
            'confidence_score': ['mean', 'std', 'count'],
            'cascade_risk': 'mean',
            'market_stress': 'mean'
        }).round(3)
        
        # Monthly analysis
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly_analysis = df.groupby('month').agg({
            'confidence_score': ['mean', 'std', 'max'],
            'price': ['first', 'last'],
            'cascade_risk': 'mean'
        }).round(3)
        
        # Calculate monthly returns
        monthly_returns = []
        for month, group in df.groupby('month'):
            first_price = group['price'].iloc[0]
            last_price = group['price'].iloc[-1]
            monthly_return = (last_price - first_price) / first_price * 100
            monthly_returns.append({
                'month': str(month),
                'return_pct': monthly_return,
                'avg_confidence': group['confidence_score'].mean()
            })
        
        self.confidence_analysis = {
            'confidence_stats': confidence_stats,
            'threshold_analysis': threshold_analysis,
            'regime_analysis': regime_analysis.to_dict(),
            'monthly_analysis': monthly_analysis.to_dict(),
            'monthly_returns': monthly_returns
        }
        
        logger.info("✅ Confidence analysis completed")
    
    def generate_recommendations(self):
        """Generate confidence threshold recommendations"""
        logger.info("🎯 Generating optimal threshold recommendations...")
        
        if not self.confidence_analysis:
            logger.error("❌ No analysis data available")
            return
        
        stats = self.confidence_analysis['confidence_stats']
        threshold_data = self.confidence_analysis['threshold_analysis']
        
        print("\n" + "="*80)
        print("🎯 NANPIN BOT CONFIDENCE THRESHOLD BACKTEST RESULTS")
        print("="*80)
        
        print(f"\n📊 CONFIDENCE STATISTICS (Past 4 Months)")
        print(f"   Total Trading Days: {stats['total_days']}")
        print(f"   Average Confidence: {stats['avg_confidence']:.1%}")
        print(f"   Median Confidence: {stats['median_confidence']:.1%}")
        print(f"   Standard Deviation: {stats['std_confidence']:.1%}")
        print(f"   Range: {stats['min_confidence']:.1%} - {stats['max_confidence']:.1%}")
        
        print(f"\n📈 CONFIDENCE PERCENTILES")
        for pct, value in stats['percentiles'].items():
            print(f"   {pct} percentile: {value:.1%}")
        
        print(f"\n🎯 THRESHOLD ANALYSIS")
        print("   Threshold | Days | % Days | Buy Ops | Trades/Month | Avg Confidence")
        print("   " + "-"*70)
        
        best_threshold = None
        best_score = 0
        
        for threshold, data in threshold_data.items():
            trades_per_month = data['trades_per_month']
            pct_days = data['percentage_of_days']
            avg_conf = data['avg_confidence']
            
            # Score = balance between trading frequency and quality
            # Target: 2-8 trades per month with good confidence
            if 2 <= trades_per_month <= 8:
                score = trades_per_month * (avg_conf ** 2) * (pct_days / 100)
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            print(f"   {threshold:.0%}     | {data['qualifying_days']:4d} | {pct_days:5.1f}% | {data['buy_opportunities']:7d} | {trades_per_month:10.1f} | {avg_conf:11.1%}")
        
        print(f"\n🏆 OPTIMAL THRESHOLD RECOMMENDATION")
        if best_threshold:
            optimal_data = threshold_data[best_threshold]
            print(f"   Recommended Threshold: {best_threshold:.0%}")
            print(f"   Expected Trades/Month: {optimal_data['trades_per_month']:.1f}")
            print(f"   Trading Days: {optimal_data['percentage_of_days']:.1f}% of time")
            print(f"   Average Confidence: {optimal_data['avg_confidence']:.1%}")
        else:
            print("   ⚠️ No optimal threshold found in 2-8 trades/month range")
            print("   Consider using 50-60% threshold for moderate trading frequency")
        
        # Monthly performance analysis
        print(f"\n📅 MONTHLY PERFORMANCE ANALYSIS")
        monthly_returns = self.confidence_analysis['monthly_returns']
        
        print("   Month    | BTC Return | Avg Confidence")
        print("   " + "-"*40)
        for month_data in monthly_returns:
            print(f"   {month_data['month']} | {month_data['return_pct']:8.1f}% | {month_data['avg_confidence']:12.1%}")
        
        # Calculate correlation between confidence and returns
        confidences = [m['avg_confidence'] for m in monthly_returns]
        returns = [m['return_pct'] for m in monthly_returns]
        correlation = np.corrcoef(confidences, returns)[0, 1] if len(confidences) > 1 else 0
        
        print(f"\n📊 CONFIDENCE-RETURN CORRELATION: {correlation:.3f}")
        if correlation > 0.3:
            print("   ✅ Positive correlation - higher confidence periods show better returns")
        elif correlation < -0.3:
            print("   ⚠️ Negative correlation - review strategy effectiveness")
        else:
            print("   📊 Weak correlation - confidence may not predict returns strongly")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. Use {best_threshold:.0%} confidence threshold for optimal balance")
        print(f"   2. Expect {optimal_data['trades_per_month']:.1f} trades per month on average")
        print(f"   3. Monitor macro regime changes for confidence adjustments")
        print(f"   4. Current bot setting (65%) may be appropriate based on this analysis")
        
        print("="*80)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"backtest_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.confidence_analysis, f, indent=2, default=str)
        
        logger.info(f"📄 Results saved to {results_file}")

async def main():
    """Run confidence backtesting"""
    backtester = ConfidenceBacktester()
    
    try:
        await backtester.initialize_components()
        await backtester.run_historical_backtest()
        backtester.generate_recommendations()
        
    except Exception as e:
        logger.error(f"❌ Backtesting failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if backtester.backpack_client and hasattr(backtester.backpack_client, 'session'):
            if backtester.backpack_client.session:
                await backtester.backpack_client.session.close()

if __name__ == "__main__":
    asyncio.run(main())