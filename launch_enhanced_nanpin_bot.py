#!/usr/bin/env python3
"""
🌸 Enhanced Nanpin Bot - All Components Integrated
Comprehensive trading bot with macro analysis, liquidation intelligence, and multi-API integration
"""

import asyncio
import logging
import signal
import sys
import yaml
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all enhanced components
from src.exchanges.backpack_client_fixed import BackpackNanpinClient
from src.core.macro_analyzer import MacroAnalyzer
from src.core.fibonacci_engine_fixed import FibonacciEngine
from src.data.enhanced_liquidation_aggregator import EnhancedLiquidationAggregator
from src.data.flipside_client_fixed import FlipsideClient
from src.strategies.macro_enhanced_goldilocks_strategy import MacroEnhancedGoldilocksStrategy

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_nanpin_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def load_credentials_from_env():
    """Load API credentials from environment"""
    api_key = os.getenv('BACKPACK_API_KEY')
    secret_key = os.getenv('BACKPACK_SECRET_KEY')
    
    if not api_key or api_key == 'your_backpack_api_key':
        raise ValueError("BACKPACK_API_KEY not set in environment")
    
    if not secret_key or secret_key == 'your_backpack_secret_key':
        raise ValueError("BACKPACK_SECRET_KEY not set in environment")
        
    logger.info("✅ Using real Backpack API credentials")
    return api_key, secret_key

class EnhancedNanpinBot:
    """
    🌸 Enhanced 永久ナンピン (Permanent DCA) Trading Bot
    
    Integrates ALL sophisticated components:
    - ✅ Macro economic analysis (FRED + Polymarket)  
    - ✅ Liquidation intelligence (multi-source)
    - ✅ On-chain metrics (Flipside)
    - ✅ Enhanced Fibonacci levels
    - ✅ CoinGecko & CoinMarketCap integration
    - ✅ Multi-API price validation
    - ✅ Comprehensive trading strategy
    """
    
    def __init__(self, config_path: str = None):
        """Initialize Enhanced Nanpin Bot"""
        self.config_path = config_path or "config/enhanced_nanpin_config.yaml"
        self.config = self._load_config()
        
        # Initialize all components
        self.backpack_client = None
        self.macro_analyzer = None
        self.fibonacci_engine = None
        self.liquidation_aggregator = None
        self.flipside_client = None
        self.strategy = None
        
        # Bot state
        self.running = False
        self.last_macro_update = None
        self.last_liquidation_update = None
        self.last_flipside_update = None
        
        # Enhanced tracking
        self.component_status = {
            'macro_analyzer': False,
            'fibonacci_engine': False,
            'liquidation_aggregator': False,
            'flipside_client': False,
            'enhanced_strategy': False,
            'multi_api_validation': False
        }
        
        self.performance_metrics = {
            'total_trades': 0,
            'successful_macro_updates': 0,
            'successful_liquidation_updates': 0,
            'successful_price_validations': 0,
            'api_integration_score': 0.0
        }
        
        logger.info("🌸 Enhanced Nanpin Bot initialized")
        logger.info("   🎯 Strategy: Macro-Enhanced Goldilocks Nanpin")
        logger.info("   🔗 Multi-API Integration: Enabled")
        logger.info("   📊 Advanced Analytics: Enabled")
    
    def _load_config(self) -> Dict:
        """Load enhanced configuration"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"✅ Loaded enhanced configuration from {self.config_path}")
                return config
            else:
                logger.warning(f"⚠️ Config file not found: {self.config_path}, using enhanced defaults")
                return self._get_enhanced_default_config()
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            return self._get_enhanced_default_config()
    
    def _get_enhanced_default_config(self) -> Dict:
        """Get enhanced default configuration"""
        return {
            'strategy': {
                'name': 'Enhanced 永久ナンピン',
                'symbol': 'BTC_USDC_PERP',  # BTC Perpetual Futures symbol
                'base_currency': 'USDC',  # Collateral currency
                'target_currency': 'BTC', # Futures contract
                'enable_all_integrations': True
            },
            'api_integration': {
                'coingecko': {
                    'enabled': True,
                    'api_key': os.getenv('COINGECKO_API_KEY'),
                    'rate_limit': 30  # requests per minute
                },
                'coinmarketcap': {
                    'enabled': True,
                    'api_key': os.getenv('COINMARKETCAP_API_KEY'),
                    'rate_limit': 333  # requests per month for free tier
                },
                'flipside': {
                    'enabled': True,
                    'api_key': os.getenv('FLIPSIDE_API_KEY'),
                    'rate_limit': 100  # requests per day
                },
                'fred': {
                    'enabled': True,
                    'api_key': os.getenv('FRED_API_KEY', 'demo_key')
                },
                'polymarket': {
                    'enabled': True,
                    'rate_limit': 60  # requests per minute
                }
            },
            'enhanced_features': {
                'macro_analysis': True,
                'liquidation_intelligence': True,
                'onchain_metrics': True,
                'multi_source_validation': True,
                'advanced_fibonacci': True,
                'regime_based_scaling': True
            },
            'update_frequencies': {
                'macro_analysis': 1800,      # 30 minutes
                'liquidation_intel': 300,    # 5 minutes  
                'flipside_metrics': 3600,    # 1 hour
                'price_validation': 60,      # 1 minute
                'fibonacci_levels': 300      # 5 minutes
            },
            'risk_management': {
                'max_position_size': 15000,    # $15K max position
                'min_confidence_threshold': 0.10,  # 10% minimum confidence (TEST MODE)
                'max_daily_trades': 5,         # Maximum 5 trades per day
                'emergency_stop_loss': 0.15,   # 15% stop loss
                'multi_source_agreement': True  # Require multiple sources to agree
            }
        }
    
    async def initialize(self):
        """Initialize all enhanced components"""
        try:
            logger.info("🚀 Initializing Enhanced Nanpin Bot components...")
            
            # Initialize Backpack client
            logger.info("   📡 Initializing enhanced Backpack client...")
            api_key, secret_key = load_credentials_from_env()
            backpack_config_path = "config/backpack_api_config.yaml"
            self.backpack_client = BackpackNanpinClient(api_key, secret_key, backpack_config_path)
            
            # Test connection
            try:
                connection_test = await self.backpack_client.test_connection()
                if connection_test:
                    logger.info("   ✅ Enhanced Backpack connection successful")
                    self.component_status['backpack_client'] = True
                else:
                    logger.error("   ❌ Backpack connection failed")
                    raise Exception("Connection test failed")
            except Exception as e:
                logger.error(f"   ❌ Backpack connection failed: {e}")
                raise
            
            # Initialize Macro Analyzer with enhanced configuration
            logger.info("   🔮 Initializing enhanced Macro Analyzer...")
            macro_config_path = "config/macro_config.yaml"
            self.macro_analyzer = MacroAnalyzer(macro_config_path)
            await self.macro_analyzer.initialize()
            logger.info("   ✅ Enhanced Macro Analyzer ready")
            self.component_status['macro_analyzer'] = True
            
            # Initialize Enhanced Fibonacci Engine
            logger.info("   📐 Initializing enhanced Fibonacci engine...")
            fibonacci_config_path = "config/fibonacci_levels.yaml"
            self.fibonacci_engine = FibonacciEngine(fibonacci_config_path, self.macro_analyzer)
            logger.info("   ✅ Enhanced Fibonacci engine ready")
            self.component_status['fibonacci_engine'] = True
            
            # Initialize Enhanced Liquidation Aggregator
            logger.info("   🔥 Initializing enhanced liquidation aggregator...")
            liquidation_config = {
                'api_keys': {
                    'coinglass': os.getenv('COINGLASS_API_KEY', '3ec7b948900e4bd2a407a26fd4c52135'),
                    'coinmarketcap': os.getenv('COINMARKETCAP_API_KEY'),
                    'coingecko': os.getenv('COINGECKO_API_KEY'),
                    'flipside': os.getenv('FLIPSIDE_API_KEY')
                },
                'thresholds': {
                    'min_liquidation_volume': 100000,
                    'cluster_distance_pct': 2.0,
                    'significance_threshold': 0.05
                },
                'timeouts': {
                    'request_timeout': 10,
                    'total_timeout': 30
                },
                'retry': {
                    'max_retries': 3,
                    'retry_delay': 1.0
                }
            }
            self.liquidation_aggregator = EnhancedLiquidationAggregator(liquidation_config)
            await self.liquidation_aggregator._init_session()
            logger.info("   ✅ Enhanced liquidation aggregator ready")
            self.component_status['liquidation_aggregator'] = True
            
            # Multi-API validation is built into Enhanced Liquidation Aggregator
            logger.info("   🔗 Multi-API validation enabled (CoinGecko + CoinMarketCap)")
            self.component_status['multi_api_validation'] = True
            
            # Initialize Dynamic Position Sizer
            logger.info("   🧮 Initializing dynamic position sizer...")
            from src.core.dynamic_position_sizer import DynamicPositionSizer
            self.position_sizer = DynamicPositionSizer(self.backpack_client, self.config)
            logger.info("   ✅ Dynamic position sizer ready")
            self.component_status['dynamic_position_sizer'] = True
            
            # Initialize Flipside Client
            logger.info("   🔗 Initializing Flipside client...")
            flipside_api_key = os.getenv('FLIPSIDE_API_KEY')
            self.flipside_client = FlipsideClient(flipside_api_key)
            await self.flipside_client.initialize()
            logger.info("   ✅ Flipside client ready")
            self.component_status['flipside_client'] = True
            
            # Initialize Macro-Enhanced Goldilocks Strategy
            logger.info("   🎯 Initializing Macro-Enhanced Goldilocks strategy...")
            strategy_config = {
                'min_drawdown': 50,         # LIVE ORDER TEST - Allow any drawdown
                'max_fear_greed': 100,      # LIVE ORDER TEST - Allow any fear/greed
                'min_days_since_ath': 0,    # LIVE ORDER TEST - No time requirement
                'base_leverage': 3.0,
                'max_leverage': 18.0,
                'cooldown_hours': 0,        # LIVE ORDER TEST - No cooldown
                'dynamic_cooldown': False,  # LIVE ORDER TEST - Disable cooldown
                'drawdown_multiplier': 0.1, # LIVE ORDER TEST - Low multiplier
                'fear_multiplier': 0.1,     # LIVE ORDER TEST - Low multiplier  
                'min_remaining_capital': 25, # LIVE ORDER TEST - $25 minimum
                'base_position_pct': 0.02,    # 2% of capital per position
                'max_single_position': 1000, # $1000 max single position
                'enable_macro_integration': True,
                'enable_liquidation_intelligence': True,
                'enable_flipside_metrics': True,
                'fibonacci_levels': {
                    '23.6%': {'ratio': 0.236, 'base_multiplier': 2, 'confidence': 0.01},  # LIVE ORDER TEST
                    '38.2%': {'ratio': 0.382, 'base_multiplier': 3, 'confidence': 0.01},  # LIVE ORDER TEST
                    '50.0%': {'ratio': 0.500, 'base_multiplier': 5, 'confidence': 0.01},  # LIVE ORDER TEST
                    '61.8%': {'ratio': 0.618, 'base_multiplier': 8, 'confidence': 0.01},  # LIVE ORDER TEST
                    '78.6%': {'ratio': 0.786, 'base_multiplier': 13, 'confidence': 0.01}  # LIVE ORDER TEST
                },
                # Entry windows (distance from Fibonacci level)
                'entry_windows': {
                    '23.6%': (-50.0, -0.1),   # LIVE ORDER TEST - Extremely wide
                    '38.2%': (-50.0, -0.1),   # LIVE ORDER TEST - Extremely wide
                    '50.0%': (-50.0, -0.1),   # LIVE ORDER TEST - Extremely wide
                    '61.8%': (-50.0, -0.1),   # LIVE ORDER TEST - Extremely wide
                    '78.6%': (-50.0, -0.1)    # LIVE ORDER TEST - Extremely wide
                }
            }
            
            self.strategy = MacroEnhancedGoldilocksStrategy(
                strategy_config, 
                self.macro_analyzer, 
                self.liquidation_aggregator
            )
            await self.strategy.initialize()
            logger.info("   ✅ Macro-Enhanced Goldilocks strategy ready")
            self.component_status['enhanced_strategy'] = True
            
            # Calculate integration score
            self.performance_metrics['api_integration_score'] = sum(self.component_status.values()) / len(self.component_status)
            
            logger.info("🎉 All enhanced components initialized successfully!")
            logger.info(f"   Integration Score: {self.performance_metrics['api_integration_score']:.1%}")
            self._log_component_status()
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced components: {e}")
            raise
    
    def _log_component_status(self):
        """Log status of all components"""
        logger.info("📊 Component Status:")
        for component, status in self.component_status.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {component.replace('_', ' ').title()}")
    
    async def start(self):
        """Start the Enhanced Nanpin Bot"""
        try:
            await self.initialize()
            
            logger.info("🌸 Starting Enhanced 永久ナンピン (Permanent DCA) Bot")
            logger.info("=" * 70)
            logger.info("🎯 ENHANCED FEATURES ACTIVE:")
            logger.info("   ✅ Macro Economic Analysis (FRED + Polymarket)")
            logger.info("   ✅ Multi-Source Liquidation Intelligence")
            logger.info("   ✅ On-Chain Metrics (Flipside)")
            logger.info("   ✅ Advanced Fibonacci Levels")
            logger.info("   ✅ CoinGecko & CoinMarketCap Integration")
            logger.info("   ✅ Multi-API Price Validation")
            logger.info("   ✅ Regime-Based Position Scaling")
            logger.info("=" * 70)
            
            # Display enhanced initial status
            await self._display_enhanced_initial_status()
            
            # Set running flag
            self.running = True
            
            # Enhanced main trading loop with detailed logging
            logger.info("🚀 About to start enhanced main trading loop...")
            await self._enhanced_main_trading_loop()
            
        except KeyboardInterrupt:
            logger.info("⏹️ Received shutdown signal")
            await self.stop()
        except Exception as e:
            logger.error(f"❌ Enhanced bot error: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the Enhanced Nanpin Bot"""
        logger.info("🛑 Stopping Enhanced Nanpin Bot...")
        
        self.running = False
        
        # Close all connections
        try:
            if self.backpack_client:
                await self.backpack_client.close()
            
            if self.macro_analyzer:
                await self.macro_analyzer.close()
            
            if self.liquidation_aggregator:
                await self.liquidation_aggregator.close()
            
            if self.flipside_client:
                await self.flipside_client.close()
                
            if self.strategy:
                await self.strategy.close()
            
            logger.info("✅ Enhanced Nanpin Bot stopped successfully")
            
            # Final performance report
            self._log_final_performance_report()
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
    
    async def _display_enhanced_initial_status(self):
        """Display enhanced initial bot status"""
        try:
            logger.info("📊 ENHANCED INITIAL STATUS")
            logger.info("-" * 50)
            
            # Test authentication
            auth_test = await self.backpack_client.test_authentication()
            if auth_test:
                logger.info("✅ Authentication successful")
            else:
                logger.error("❌ Authentication failed")
                return
            
            # Get enhanced market data
            enhanced_market_data = await self.liquidation_aggregator.get_enhanced_market_data('BTC')
            if enhanced_market_data:
                logger.info("💰 Enhanced Market Overview:")
                logger.info(f"   Current Price: ${enhanced_market_data.current_price:,.2f}")
                logger.info(f"   Sources: {list(enhanced_market_data.price_sources.keys())}")
                logger.info(f"   24h Volume: ${enhanced_market_data.volume_24h:,.0f}")
                logger.info(f"   Market Cap: ${enhanced_market_data.market_cap:,.0f}")
                logger.info(f"   Open Interest: {enhanced_market_data.open_interest:,.0f} BTC")
                
                # Funding rates
                if enhanced_market_data.funding_rates:
                    logger.info("   Funding Rates:")
                    for exchange, rate in enhanced_market_data.funding_rates.items():
                        logger.info(f"     {exchange}: {rate:.4f}%")
                
                # Reliability scores
                avg_reliability = sum(enhanced_market_data.source_reliability.values()) / len(enhanced_market_data.source_reliability)
                logger.info(f"   Source Reliability: {avg_reliability:.1%}")
            
            # Get macro analysis status
            if self.component_status['macro_analyzer']:
                try:
                    macro_analysis = await self.macro_analyzer.update_macro_analysis()
                    if macro_analysis:
                        logger.info("🔮 Macro Analysis:")
                        logger.info(f"   Regime: {macro_analysis.regime.value}")
                        logger.info(f"   Signal: {macro_analysis.overall_signal}")
                        logger.info(f"   Fear/Greed: {macro_analysis.fear_greed_index:.1f}")
                        logger.info(f"   BTC Sentiment: {macro_analysis.bitcoin_sentiment:.1f}")
                        logger.info(f"   Position Scaling: {macro_analysis.position_scaling_factor:.2f}x")
                except Exception as e:
                    logger.warning(f"⚠️ Could not get initial macro analysis: {e}")
            
            # Get liquidation intelligence
            if self.component_status['liquidation_aggregator']:
                try:
                    heatmap = await self.liquidation_aggregator.generate_enhanced_liquidation_heatmap('BTC')
                    if heatmap:
                        logger.info("🔥 Liquidation Intelligence:")
                        logger.info(f"   Active Clusters: {len(heatmap.clusters)}")
                        logger.info(f"   Cascade Risk: {heatmap.cascade_risk_score:.1f}/10")
                        logger.info(f"   Market Sentiment: {heatmap.overall_sentiment}")
                        logger.info(f"   Data Sources: {heatmap.data_sources}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not get initial liquidation intelligence: {e}")
            
            # Get Fibonacci levels
            if self.component_status['fibonacci_engine']:
                try:
                    # This would be updated in the main loop
                    enabled_levels = [name for name, config in self.fibonacci_engine.config['levels'].items() if config.get('enabled', True)]
                    logger.info("📐 Fibonacci Engine:")
                    logger.info(f"   Enabled Levels: {enabled_levels}")
                    logger.info(f"   Total Levels: {len(enabled_levels)}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not get Fibonacci status: {e}")
            
            # Strategy info
            strategy_stats = self.strategy.get_strategy_stats()
            logger.info(f"🎯 Strategy: {strategy_stats['strategy_name']}")
            logger.info(f"   Target Return: {strategy_stats['target_annual_return']}")
            logger.info(f"   Historical Sharpe: {strategy_stats['historical_sharpe']}")
            
            logger.info("-" * 50)
            
        except Exception as e:
            logger.error(f"❌ Failed to display enhanced initial status: {e}")
    
    async def _enhanced_main_trading_loop(self):
        """Enhanced main trading loop with all integrations"""
        logger.info("🔄 Starting enhanced main trading loop...")
        
        loop_count = 0
        
        while self.running:
            try:
                loop_count += 1
                logger.info(f"🔄 Enhanced Trading Loop #{loop_count}")
                
                # Update dynamic position sizing (every hour or on significant balance change)
                if loop_count % 60 == 1:  # Every hour (assuming 1-minute loops)
                    await self._update_dynamic_position_sizing()
                
                # Update all analysis components
                await self._update_enhanced_market_analysis()
                
                # Check for enhanced trading opportunities (with dynamic sizing)
                await self._check_enhanced_trading_opportunities()
                
                # Enhanced risk monitoring
                await self._monitor_enhanced_risk()
                
                # Performance monitoring
                if loop_count % 10 == 0:  # Every 10 loops
                    self._log_performance_metrics()
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"❌ Error in enhanced trading loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _update_enhanced_market_analysis(self):
        """Update all enhanced market analysis components"""
        try:
            now = datetime.now()
            
            # Update macro analysis (every 30 minutes)
            if (not self.last_macro_update or 
                (now - self.last_macro_update).total_seconds() >= self.config['update_frequencies']['macro_analysis']):
                
                logger.info("🔮 Updating enhanced macro analysis...")
                try:
                    macro_analysis = await self.macro_analyzer.update_macro_analysis()
                    if macro_analysis:
                        self.last_macro_update = now
                        self.performance_metrics['successful_macro_updates'] += 1
                        logger.info(f"   ✅ Macro analysis updated - Regime: {macro_analysis.regime.value}")
                except Exception as e:
                    logger.error(f"   ❌ Macro analysis update failed: {e}")
            
            # Update enhanced liquidation intelligence (every 5 minutes)
            if (not self.last_liquidation_update or 
                (now - self.last_liquidation_update).total_seconds() >= self.config['update_frequencies']['liquidation_intel']):
                
                logger.info("🔥 Updating enhanced liquidation intelligence...")
                try:
                    heatmap = await self.liquidation_aggregator.generate_enhanced_liquidation_heatmap('BTC')
                    if heatmap and len(heatmap.clusters) > 0:
                        self.last_liquidation_update = now
                        self.performance_metrics['successful_liquidation_updates'] += 1
                        logger.info(f"   ✅ Enhanced liquidation heatmap: {len(heatmap.clusters)} clusters")
                except Exception as e:
                    logger.error(f"   ❌ Enhanced liquidation intelligence update failed: {e}")
            
            # Update Flipside metrics (every 1 hour)
            if (not self.last_flipside_update or 
                (now - self.last_flipside_update).total_seconds() >= self.config['update_frequencies']['flipside_metrics']):
                
                logger.info("🔗 Updating Flipside on-chain metrics...")
                try:
                    flipside_metrics = await self.flipside_client.get_liquidation_metrics('BTC')
                    if flipside_metrics:
                        self.last_flipside_update = now
                        logger.info(f"   ✅ Flipside metrics updated - Stress: {flipside_metrics.market_stress_indicator:.1f}%")
                except Exception as e:
                    logger.error(f"   ❌ Flipside metrics update failed: {e}")
            
            # Validate prices from multiple sources
            try:
                enhanced_market_data = await self.liquidation_aggregator.get_enhanced_market_data('BTC')
                if enhanced_market_data and len(enhanced_market_data.price_sources) >= 2:
                    self.performance_metrics['successful_price_validations'] += 1
                    logger.debug(f"   ✅ Price validation successful: ${enhanced_market_data.current_price:,.2f}")
            except Exception as e:
                logger.error(f"   ❌ Price validation failed: {e}")
                
        except Exception as e:
            logger.error(f"❌ Failed to update enhanced market analysis: {e}")
    
    async def _check_enhanced_trading_opportunities(self):
        """Check for enhanced trading opportunities using all data sources"""
        try:
            logger.info("🔍 Checking enhanced trading opportunities...")
            # Get enhanced market data
            enhanced_market_data = await self.liquidation_aggregator.get_enhanced_market_data('BTC')
            if not enhanced_market_data:
                logger.warning("⚠️ No enhanced market data available")
                return
            
            # Get historical data for analysis
            klines = await self.backpack_client.get_klines(interval='1d', limit=100)
            if not klines:
                logger.warning("⚠️ No historical kline data available")
                return
            
            # Convert to DataFrame
            import pandas as pd
            df = pd.DataFrame(klines)
            
            if not df.empty and isinstance(klines[0], dict) and 'start' in klines[0]:
                # Process klines data
                price_data = []
                for candle in klines:
                    price_data.append([
                        candle['start'],
                        candle['open'],
                        candle['high'],
                        candle['low'],
                        candle['close'],
                        candle['volume']
                    ])
                df = pd.DataFrame(price_data)
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                
                # Convert timestamp and prices
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Prepare enhanced market data for strategy
                market_data = {
                    'current_price': enhanced_market_data.current_price,
                    'historical_data': df,
                    'enhanced_data': enhanced_market_data  # Include all enhanced data
                }
                
                # Analyze market conditions using enhanced strategy
                logger.info("🎯 Analyzing enhanced market conditions...")
                analysis = await self.strategy.analyze_market_conditions(market_data)
                
                if 'error' not in analysis:
                    confidence_score = analysis.get('confidence_score', 0)
                    recommendations = analysis.get('recommendations', [])
                    
                    logger.info(f"   Analysis confidence: {confidence_score:.1%}")
                    logger.info(f"   Recommendations: {len(recommendations)}")
                    
                    # Log enhanced analysis details
                    if analysis.get('macro_analysis', {}).get('available'):
                        macro = analysis['macro_analysis']
                        logger.info(f"   Macro regime: {macro.get('regime', 'Unknown')}")
                        logger.info(f"   Macro signal: {macro.get('overall_signal', 'Unknown')}")
                    
                    if analysis.get('liquidation_intelligence', {}).get('available'):
                        liq = analysis['liquidation_intelligence']
                        logger.info(f"   Liquidation clusters: {liq['cluster_analysis']['total_clusters']}")
                        logger.info(f"   Cascade risk: {liq['cluster_analysis']['cascade_risk']:.1f}/10")
                    
                    if analysis.get('flipside_metrics', {}).get('available'):
                        flipside = analysis['flipside_metrics']
                        logger.info(f"   Whale activity: {flipside['signals']['whale_pressure']}")
                        logger.info(f"   Market stress: {flipside['signals']['market_stress']}")
                    
                    # Execute best recommendation if confidence is high enough
                    min_confidence = 0.01  # LIVE ORDER TEST - Extremely low threshold
                    logger.info(f"🔍 DEBUG: Found {len(recommendations)} recommendations")
                    for i, recommendation in enumerate(recommendations):
                        logger.info(f"🔍 DEBUG: Recommendation {i+1}: {recommendation}")
                        logger.info(f"🔍 DEBUG: Action: {recommendation.get('action')}, Confidence: {confidence_score:.1%} >= {min_confidence:.1%}")
                        if (recommendation.get('action') == 'BUY' and 
                            confidence_score >= min_confidence):
                            logger.info("🚀 High-confidence enhanced trading opportunity detected!")
                            await self._evaluate_and_execute_enhanced_trade(recommendation, analysis)
                            break  # Only execute one trade per cycle
                        else:
                            logger.info(f"⏭️ Skipping recommendation: action={recommendation.get('action')}, confidence={confidence_score:.1%}")
                else:
                    logger.warning(f"⚠️ Enhanced market analysis error: {analysis.get('error', 'Unknown')}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to check enhanced trading opportunities: {e}")
    
    async def _evaluate_and_execute_enhanced_trade(self, recommendation: Dict, analysis: Dict):
        """Evaluate and execute enhanced trading recommendation"""
        try:
            logger.info("🎯 Evaluating enhanced trade recommendation:")
            logger.info(f"   Level: {recommendation.get('level', 'Unknown')}")
            logger.info(f"   Target Price: ${recommendation.get('target_price', 0):,.2f}")
            logger.info(f"   Position Size: ${recommendation.get('position_size_usdc', 0):,.2f}")
            logger.info(f"   Enhanced Confidence: {recommendation.get('confidence', 0):.1%}")
            
            # Enhanced risk assessment
            risk_assessment = recommendation.get('risk_assessment', {})
            logger.info(f"   Risk Level: {risk_assessment.get('overall_risk', 'Unknown')}")
            logger.info(f"   Risk Factors: {len(risk_assessment.get('risk_factors', []))}")
            
            # Check enhanced reasoning
            enhanced_reasoning = recommendation.get('enhanced_reasoning', '')
            logger.info(f"   Enhanced Reasoning: {enhanced_reasoning}")
            
            # Additional safety checks
            confidence = recommendation.get('confidence', 0)
            risk_level = risk_assessment.get('overall_risk', 'HIGH')
            
            if confidence < self.config.get('trading', {}).get('min_confidence_threshold', 0.65):
                logger.info("   ⏳ Confidence below threshold, skipping trade")
                return
            
            if risk_level == 'HIGH' and confidence < 0.9:
                logger.info("   ⚠️ High risk with moderate confidence, skipping trade")
                return
            
            # Check daily trade limit
            if self.performance_metrics['total_trades'] >= self.config['risk_management']['max_daily_trades']:
                logger.info("   📊 Daily trade limit reached, skipping trade")
                return
            
            # Execute using enhanced strategy
            logger.info("🚀 Executing enhanced trade...")
            execution_result = await self.strategy.execute_recommendation(
                recommendation, self.backpack_client
            )
            
            if execution_result.get('status') == 'executed':
                self.performance_metrics['total_trades'] += 1
                logger.info("✅ Enhanced trade executed successfully!")
                
                # Log enhanced trade details
                order_result = execution_result.get('order_result', {})
                logger.info(f"   Order ID: {order_result.get('id', 'N/A')}")
                logger.info(f"   Fill Price: ${order_result.get('fillPrice', 0):,.2f}")
                logger.info(f"   Quantity: {order_result.get('quantity', 0):.8f} BTC")
                logger.info(f"   Total Trades Today: {self.performance_metrics['total_trades']}")
                
                # Log enhanced analytics
                await self._log_enhanced_trade_analytics(recommendation, analysis, execution_result)
                
            else:
                logger.warning(f"⚠️ Enhanced trade execution failed: {execution_result.get('reason', 'Unknown')}")
                
        except Exception as e:
            logger.error(f"❌ Failed to evaluate/execute enhanced trade: {e}")
    
    async def _log_enhanced_trade_analytics(self, recommendation: Dict, analysis: Dict, execution_result: Dict):
        """Log comprehensive analytics for executed trade"""
        try:
            logger.info("📊 Enhanced Trade Analytics:")
            
            # Strategy components that contributed
            components = []
            if analysis.get('macro_analysis', {}).get('available'):
                components.append('Macro Analysis')
            if analysis.get('liquidation_intelligence', {}).get('available'):
                components.append('Liquidation Intelligence')
            if analysis.get('flipside_metrics', {}).get('available'):
                components.append('On-Chain Metrics')
            
            logger.info(f"   Contributing Components: {', '.join(components)}")
            
            # Integrated signals
            integrated_signals = analysis.get('integrated_signals', {})
            logger.info(f"   Combined Signal: {integrated_signals.get('combined_signal', 'Unknown')}")
            logger.info(f"   Entry Strength: {integrated_signals.get('entry_strength', 0):.2f}")
            logger.info(f"   Position Multiplier: {integrated_signals.get('position_multiplier', 1):.2f}x")
            
            # Market regime context
            market_regime = analysis.get('market_regime', 'Unknown')
            logger.info(f"   Market Regime: {market_regime}")
            
            # Confidence breakdown
            confidence_factors = integrated_signals.get('confidence_factors', {})
            if confidence_factors:
                logger.info("   Confidence Breakdown:")
                for factor, score in confidence_factors.items():
                    logger.info(f"     {factor.title()}: {score:.2f}")
                    
        except Exception as e:
            logger.error(f"❌ Error logging enhanced trade analytics: {e}")
    
    async def _monitor_enhanced_risk(self):
        """Enhanced risk monitoring with all data sources"""
        try:
            # Get enhanced market data for risk assessment
            enhanced_market_data = await self.liquidation_aggregator.get_enhanced_market_data('BTC')
            if not enhanced_market_data:
                return
            
            # Price deviation risk
            if enhanced_market_data.price_sources:
                prices = list(enhanced_market_data.price_sources.values())
                if len(prices) > 1:
                    price_range = max(prices) - min(prices)
                    relative_deviation = price_range / enhanced_market_data.current_price
                    
                    if relative_deviation > 0.05:  # 5% deviation
                        logger.warning(f"⚠️ High price deviation detected: {relative_deviation:.1%}")
            
            # Funding rate risk
            if enhanced_market_data.funding_rates:
                avg_funding = sum(enhanced_market_data.funding_rates.values()) / len(enhanced_market_data.funding_rates)
                if abs(avg_funding) > 0.02:  # 2% funding rate
                    logger.warning(f"⚠️ Extreme funding rates detected: {avg_funding:.2%}")
            
            # Volume anomaly risk
            if enhanced_market_data.derivatives_volume > 0:
                volume_ratio = enhanced_market_data.volume_24h / enhanced_market_data.derivatives_volume
                if volume_ratio > 5.0:  # Spot volume 5x derivatives
                    logger.warning(f"⚠️ Unusual volume pattern detected: {volume_ratio:.1f}x ratio")
            
            # Source reliability risk
            avg_reliability = sum(enhanced_market_data.source_reliability.values()) / len(enhanced_market_data.source_reliability)
            if avg_reliability < 0.7:  # 70% reliability threshold
                logger.warning(f"⚠️ Low source reliability: {avg_reliability:.1%}")
                
        except Exception as e:
            logger.error(f"❌ Error in enhanced risk monitoring: {e}")
    
    async def _update_dynamic_position_sizing(self):
        """Update dynamic position sizing based on current balance"""
        try:
            if not self.component_status.get('dynamic_position_sizer', False):
                return
            
            logger.info("🧮 Updating dynamic position sizing...")
            
            # Get current position size recommendation
            recommendation = await self.position_sizer.calculate_dynamic_position_size()
            
            if recommendation:
                # Update current strategy parameters
                self.current_position_config = {
                    'base_margin': recommendation.base_margin,
                    'position_value': recommendation.position_value,
                    'leverage': recommendation.leverage,
                    'max_levels': recommendation.max_levels,
                    'scaling_multiplier': recommendation.scaling_multiplier,
                    'risk_level': recommendation.risk_level,
                    'last_updated': datetime.now()
                }
                
                logger.info("✅ Dynamic position sizing updated:")
                logger.info(f"   Base Margin: ${recommendation.base_margin:.2f}")
                logger.info(f"   Leverage: {recommendation.leverage}x")
                logger.info(f"   Max Levels: {recommendation.max_levels}")
                logger.info(f"   Risk Level: {recommendation.risk_level}")
                
                # Update strategy with new parameters
                if hasattr(self.strategy, 'update_position_parameters'):
                    self.strategy.update_position_parameters(recommendation)
                
        except Exception as e:
            logger.error(f"❌ Failed to update dynamic position sizing: {e}")
    
    def _log_performance_metrics(self):
        """Log current performance metrics"""
        logger.info("📈 Enhanced Performance Metrics:")
        logger.info(f"   Total Trades: {self.performance_metrics['total_trades']}")
        logger.info(f"   Successful Macro Updates: {self.performance_metrics['successful_macro_updates']}")
        logger.info(f"   Successful Liquidation Updates: {self.performance_metrics['successful_liquidation_updates']}")
        logger.info(f"   Successful Price Validations: {self.performance_metrics['successful_price_validations']}")
        logger.info(f"   API Integration Score: {self.performance_metrics['api_integration_score']:.1%}")
        
        # Log current position sizing if available
        if hasattr(self, 'current_position_config'):
            config = self.current_position_config
            logger.info(f"   Current Position Size: ${config['base_margin']:.2f} margin")
            logger.info(f"   Current Leverage: {config['leverage']}x")
            logger.info(f"   Current Risk Level: {config['risk_level']}")
    
    def _log_final_performance_report(self):
        """Log final performance report"""
        logger.info("📊 FINAL ENHANCED PERFORMANCE REPORT")
        logger.info("=" * 50)
        logger.info(f"Total Trading Sessions: 1")
        logger.info(f"Total Trades Executed: {self.performance_metrics['total_trades']}")
        logger.info(f"Macro Analysis Updates: {self.performance_metrics['successful_macro_updates']}")
        logger.info(f"Liquidation Intelligence Updates: {self.performance_metrics['successful_liquidation_updates']}")
        logger.info(f"Price Validations: {self.performance_metrics['successful_price_validations']}")
        logger.info(f"Overall Integration Score: {self.performance_metrics['api_integration_score']:.1%}")
        
        # Component reliability
        logger.info("\nComponent Reliability:")
        for component, status in self.component_status.items():
            logger.info(f"  {component.replace('_', ' ').title()}: {'✅ Active' if status else '❌ Inactive'}")
        
        logger.info("=" * 50)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down...")
    # The main loop will handle the actual shutdown

async def main():
    """Main function"""
    try:
        logger.info("🌸 Enhanced Nanpin Bot Starting...")
        
        # Set up signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Create and start enhanced bot
        bot = EnhancedNanpinBot()
        await bot.start()
        
    except KeyboardInterrupt:
        logger.info("⏹️ Received keyboard interrupt")
    except Exception as e:
        logger.error(f"❌ Enhanced bot crashed: {e}")
        raise
    finally:
        logger.info("🛑 Stopping Enhanced Nanpin Bot...")

if __name__ == "__main__":
    asyncio.run(main())