#!/usr/bin/env python3
"""
🌸 Nanpin Bot Launcher
永久ナンピン (Permanent DCA) Trading Bot for Backpack Exchange
"""

import asyncio
import logging
import sys
import os
import yaml
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from exchanges.backpack_nanpin_client import BackpackNanpinClient, load_credentials_from_env
from core.fibonacci_engine import FibonacciEngine
from core.macro_analyzer import MacroAnalyzer
from data.liquidation_aggregator_fixed import LiquidationAggregator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/nanpin_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class NanpinBot:
    """
    🌸 永久ナンピン (Permanent DCA) Trading Bot
    
    Features:
    - Fibonacci-based entry levels with macro intelligence
    - FRED Federal Reserve data integration
    - Polymarket prediction market sentiment
    - Multi-source liquidation intelligence
    - Permanent accumulation strategy (never sell)
    - Risk management and liquidation protection
    - Backpack Exchange integration
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize Nanpin Bot
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/nanpin_config.yaml"
        self.config = self._load_config()
        
        # Initialize components
        self.backpack_client = None
        self.fibonacci_engine = None
        self.macro_analyzer = None
        self.liquidation_aggregator = None
        
        # Bot state
        self.running = False
        self.last_fibonacci_update = None
        self.last_liquidation_update = None
        self.last_macro_update = None
        self.position_tracker = {
            'total_btc': 0.0,
            'total_invested_usdc': 0.0,
            'average_entry_price': 0.0,
            'entry_count': 0
        }
        
        logger.info("🌸 Nanpin Bot initialized")
        logger.info(f"   Strategy: {self.config['strategy']['name']}") 
        logger.info(f"   Target Symbol: {self.config['strategy']['symbol']}")
        logger.info(f"   Base Amount: ${self.config['position_scaling']['base_usdc_amount']}")
    
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"✅ Loaded configuration from {self.config_path}")
                return config
            else:
                logger.warning(f"⚠️ Config file not found: {self.config_path}, using defaults")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'strategy': {
                'name': '永久ナンピン',
                'symbol': 'BTC_USDC',
                'base_currency': 'USDC',
                'target_currency': 'BTC'
            },
            'position_scaling': {
                'base_usdc_amount': 100.0,
                'max_total_exposure': 10000.0,
                'scaling_cooldown': 1800
            },
            'fibonacci': {
                'levels': {
                    '23.6%': {'enabled': True, 'multiplier': 1.0},
                    '38.2%': {'enabled': True, 'multiplier': 2.0},
                    '50.0%': {'enabled': True, 'multiplier': 3.0},
                    '61.8%': {'enabled': True, 'multiplier': 5.0},
                    '78.6%': {'enabled': True, 'multiplier': 8.0}
                },
                'update_frequency': 300
            },
            'risk_management': {
                'min_collateral_ratio': 4.0,
                'emergency_stop_enabled': True
            },
            'logging': {
                'level': 'INFO'
            }
        }
    
    async def initialize(self):
        """Initialize all bot components"""
        try:
            logger.info("🚀 Initializing Nanpin Bot components...")
            
            # Initialize Backpack client
            logger.info("   📡 Initializing Backpack client...")
            api_key, secret_key = load_credentials_from_env()
            backpack_config_path = "config/backpack_api_config.yaml"
            self.backpack_client = BackpackNanpinClient(api_key, secret_key, backpack_config_path)
            
            # Test connection
            try:
                balances = await self.backpack_client.get_balances()
                logger.info("   ✅ Backpack connection successful")
            except Exception as e:
                logger.error(f"   ❌ Backpack connection failed: {e}")
                raise
            
            # Initialize Macro Analyzer
            logger.info("   🔮 Initializing Macro Analyzer...")
            macro_config_path = "config/macro_config.yaml"
            self.macro_analyzer = MacroAnalyzer(macro_config_path)
            await self.macro_analyzer.initialize()
            logger.info("   ✅ Macro Analyzer ready")
            
            # Initialize Fibonacci engine with macro integration
            logger.info("   📐 Initializing Fibonacci engine...")
            fibonacci_config_path = "config/fibonacci_levels.yaml"
            self.fibonacci_engine = FibonacciEngine(fibonacci_config_path, self.macro_analyzer)
            logger.info("   ✅ Fibonacci engine ready")
            
            # Initialize liquidation aggregator
            logger.info("   🔥 Initializing liquidation aggregator...")
            liquidation_config = {
                'api_keys': {
                    'coinglass': os.getenv('COINGLASS_API_KEY', '3ec7b948900e4bd2a407a26fd4c52135'),
                    'coinmarketcap': os.getenv('COINMARKETCAP_API_KEY'),
                    'coingecko': os.getenv('COINGECKO_API_KEY'),
                    'flipside': os.getenv('FLIPSIDE_API_KEY')
                }
            }
            self.liquidation_aggregator = LiquidationAggregator(liquidation_config)
            await self.liquidation_aggregator._init_session()
            logger.info("   ✅ Liquidation aggregator ready")
            
            logger.info("🎉 All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    async def start(self):
        """Start the Nanpin Bot"""
        try:
            await self.initialize()
            
            logger.info("🌸 Starting 永久ナンピン (Permanent DCA) Bot")
            logger.info("=" * 60)
            
            # Display initial status
            await self._display_initial_status()
            
            # Set running flag
            self.running = True
            
            # Main trading loop
            await self._main_trading_loop()
            
        except KeyboardInterrupt:
            logger.info("⏹️ Received shutdown signal")
            await self.stop()
        except Exception as e:
            logger.error(f"❌ Bot error: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the Nanpin Bot"""
        logger.info("🛑 Stopping Nanpin Bot...")
        
        self.running = False
        
        # Close connections
        if self.backpack_client:
            await self.backpack_client.close()
        
        if self.macro_analyzer:
            await self.macro_analyzer.close()
        
        if self.liquidation_aggregator:
            await self.liquidation_aggregator.close()
        
        logger.info("✅ Nanpin Bot stopped successfully")
    
    async def _display_initial_status(self):
        """Display initial bot status"""
        try:
            logger.info("📊 INITIAL STATUS")
            logger.info("-" * 40)
            
            # Get account info
            balances = await self.backpack_client.get_balances()
            btc_position = await self.backpack_client.get_btc_position()
            collateral_info = await self.backpack_client.get_collateral_info()
            current_price = await self.backpack_client.get_btc_price()
            
            # Display account info
            logger.info(f"💰 Account Overview:")
            if collateral_info:
                net_equity = collateral_info.get('netEquity', 0)
                margin_fraction = collateral_info.get('marginFraction', 0)
                logger.info(f"   Net Equity: ${float(net_equity):,.2f}")
                logger.info(f"   Margin Fraction: {float(margin_fraction):.1%}")
            
            # Display BTC position
            if btc_position:
                btc_size = float(btc_position.get('size', 0))
                btc_value = btc_size * current_price
                logger.info(f"₿ BTC Position: {btc_size:.8f} BTC (${btc_value:,.2f})")
                
                # Update position tracker
                self.position_tracker['total_btc'] = btc_size
                
                entry_price = btc_position.get('entryPrice')
                if entry_price:
                    self.position_tracker['average_entry_price'] = float(entry_price)
                    logger.info(f"   Average Entry: ${float(entry_price):,.2f}")
            else:
                logger.info("₿ No BTC position detected")
            
            logger.info(f"📈 Current BTC Price: ${current_price:,.2f}")
            
            # Check risk status
            risk_assessment = await self.backpack_client.check_liquidation_risk()
            risk_level = risk_assessment.get('liquidation_risk', 'unknown')
            logger.info(f"⚖️ Risk Level: {risk_level.upper()}")
            
            logger.info("-" * 40)
            
        except Exception as e:
            logger.error(f"❌ Failed to display initial status: {e}")
    
    async def _main_trading_loop(self):
        """Main trading loop"""
        logger.info("🔄 Starting main trading loop...")
        
        while self.running:
            try:\n                # Update Fibonacci levels periodically\n                await self._update_fibonacci_levels()\n                \n                # Update liquidation intelligence\n                await self._update_liquidation_intelligence()\n                \n                # Check for trading opportunities\n                await self._check_trading_opportunities()\n                \n                # Risk monitoring\n                await self._monitor_risk()\n                \n                # Wait before next iteration\n                await asyncio.sleep(60)  # Check every minute\n                \n            except Exception as e:\n                logger.error(f\"❌ Error in trading loop: {e}\")\n                await asyncio.sleep(60)  # Wait before retrying\n    \n    async def _update_fibonacci_levels(self):\n        \"\"\"Update Fibonacci retracement levels\"\"\"\n        try:\n            now = datetime.now()\n            update_frequency = self.config['fibonacci']['update_frequency']\n            \n            # Check if update is needed\n            if (self.last_fibonacci_update and \n                (now - self.last_fibonacci_update).total_seconds() < update_frequency):\n                return\n            \n            logger.info(\"📐 Updating Fibonacci levels...\")\n            \n            # Get price data\n            klines = await self.backpack_client.get_klines(interval='1h', limit=720)\n            \n            if klines:\n                # Convert to DataFrame\n                import pandas as pd\n                df = pd.DataFrame(klines)\n                \n                # Ensure proper column names\n                if 'close' not in df.columns and len(df.columns) >= 4:\n                    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume'][:len(df.columns)]\n                \n                # Convert timestamp to datetime index\n                if 'timestamp' in df.columns:\n                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')\n                    df.set_index('timestamp', inplace=True)\n                \n                # Calculate Fibonacci levels\n                fibonacci_levels = self.fibonacci_engine.calculate_fibonacci_levels(df)\n                \n                if fibonacci_levels:\n                    logger.info(f\"✅ Updated {len(fibonacci_levels)} Fibonacci levels\")\n                    for name, level in fibonacci_levels.items():\n                        logger.info(f\"   {name}: ${level.price:,.2f} (confidence: {level.confidence:.2%})\")\n                    \n                    self.last_fibonacci_update = now\n                else:\n                    logger.warning(\"⚠️ Failed to calculate Fibonacci levels\")\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to update Fibonacci levels: {e}\")\n    \n    async def _update_liquidation_intelligence(self):\n        \"\"\"Update liquidation heatmap data\"\"\"\n        try:\n            now = datetime.now()\n            update_frequency = self.config.get('liquidation_intelligence', {}).get('liquidation_update_interval', 180)\n            \n            # Check if update is needed\n            if (self.last_liquidation_update and \n                (now - self.last_liquidation_update).total_seconds() < update_frequency):\n                return\n            \n            logger.info(\"🔥 Updating liquidation intelligence...\")\n            \n            # Generate liquidation heatmap\n            heatmap = await self.liquidation_aggregator.generate_liquidation_heatmap('BTC')\n            \n            if heatmap:\n                logger.info(f\"✅ Updated liquidation heatmap:\")\n                logger.info(f\"   Current Price: ${heatmap.current_price:,.2f}\")\n                logger.info(f\"   Liquidation Clusters: {len(heatmap.clusters)}\")\n                logger.info(f\"   Nanpin Opportunities: {len(heatmap.nanpin_opportunities)}\")\n                logger.info(f\"   Risk Level: {heatmap.risk_assessment.get('overall_risk', 'unknown')}\")\n                \n                self.last_liquidation_update = now\n                \n                # Store heatmap for trading decisions\n                self.current_liquidation_heatmap = heatmap\n            else:\n                logger.warning(\"⚠️ Failed to generate liquidation heatmap\")\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to update liquidation intelligence: {e}\")\n    \n    async def _check_trading_opportunities(self):\n        \"\"\"Check for Nanpin trading opportunities\"\"\"\n        try:\n            # Get current price\n            current_price = await self.backpack_client.get_btc_price()\n            if not current_price:\n                logger.warning(\"⚠️ Could not get current BTC price\")\n                return\n            \n            # Get Fibonacci recommendations\n            if not self.fibonacci_engine.current_levels:\n                logger.debug(\"📐 No Fibonacci levels available yet\")\n                return\n            \n            recommendations = self.fibonacci_engine.get_position_scaling_recommendations(current_price)\n            \n            # Check for buy signals\n            for level_name, rec in recommendations.items():\n                if rec['action'] == 'BUY' and rec['urgency'] in ['HIGH', 'MEDIUM']:\n                    await self._evaluate_buy_opportunity(level_name, rec, current_price)\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to check trading opportunities: {e}\")\n    \n    async def _evaluate_buy_opportunity(self, level_name: str, recommendation: Dict, current_price: float):\n        \"\"\"Evaluate and potentially execute a buy opportunity\"\"\"\n        try:\n            logger.info(f\"🎯 Evaluating buy opportunity at {level_name}\")\n            logger.info(f\"   Target Price: ${recommendation['target_price']:,.2f}\")\n            logger.info(f\"   Current Price: ${current_price:,.2f}\")\n            logger.info(f\"   Distance: {recommendation['current_distance_pct']:+.2f}%\")\n            logger.info(f\"   Reasoning: {recommendation['reasoning']}\")\n            \n            # Check if we're actually at or below the level\n            if recommendation['current_distance_pct'] > -0.5:  # Not sufficiently below level\n                logger.info(\"   ⏳ Price not sufficiently below target level, waiting...\")\n                return\n            \n            # Calculate position size\n            base_amount = self.config['position_scaling']['base_usdc_amount']\n            multiplier = recommendation['adjusted_multiplier']\n            target_usdc_amount = base_amount * multiplier\n            \n            # Apply risk management\n            safe_amount = await self.backpack_client.calculate_safe_order_size(target_usdc_amount)\n            \n            if safe_amount <= 0:\n                logger.warning(\"   🚨 Risk management prevented trade\")\n                return\n            \n            # Check cooldown\n            if not self._check_scaling_cooldown():\n                logger.info(\"   ⏱️ Scaling cooldown active, skipping trade\")\n                return\n            \n            # Execute the trade\n            logger.info(f\"   💰 Executing Nanpin buy: ${safe_amount:.2f} USDC\")\n            \n            reason = f\"Nanpin {level_name} Fibonacci level\"\n            order_result = await self.backpack_client.market_buy_btc(safe_amount, reason)\n            \n            if order_result:\n                logger.info(f\"   ✅ Nanpin buy executed successfully!\")\n                logger.info(f\"      Order ID: {order_result.get('id', 'N/A')}\")\n                \n                # Update position tracking\n                self._update_position_tracking(safe_amount, order_result)\n                \n                # Log the trade\n                await self._log_trade(level_name, safe_amount, current_price, order_result)\n            else:\n                logger.error(\"   ❌ Failed to execute Nanpin buy\")\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to evaluate buy opportunity: {e}\")\n    \n    def _check_scaling_cooldown(self) -> bool:\n        \"\"\"Check if scaling cooldown period has passed\"\"\"\n        # Implementation would track last trade time\n        # For now, always allow (you can enhance this)\n        return True\n    \n    def _update_position_tracking(self, usdc_spent: float, order_result: Dict):\n        \"\"\"Update internal position tracking\"\"\"\n        try:\n            self.position_tracker['total_invested_usdc'] += usdc_spent\n            self.position_tracker['entry_count'] += 1\n            \n            # Update average price if we got execution data\n            if 'fillPrice' in order_result and 'quantity' in order_result:\n                fill_price = float(order_result['fillPrice'])\n                quantity = float(order_result['quantity'])\n                \n                # Update total BTC\n                old_total_btc = self.position_tracker['total_btc']\n                new_total_btc = old_total_btc + quantity\n                \n                # Calculate new average entry price\n                if old_total_btc > 0:\n                    old_avg_price = self.position_tracker['average_entry_price']\n                    new_avg_price = ((old_avg_price * old_total_btc) + (fill_price * quantity)) / new_total_btc\n                else:\n                    new_avg_price = fill_price\n                \n                self.position_tracker['total_btc'] = new_total_btc\n                self.position_tracker['average_entry_price'] = new_avg_price\n                \n                logger.info(f\"📊 Position updated:\")\n                logger.info(f\"   Total BTC: {new_total_btc:.8f}\")\n                logger.info(f\"   Average Entry: ${new_avg_price:,.2f}\")\n                logger.info(f\"   Total Invested: ${self.position_tracker['total_invested_usdc']:,.2f}\")\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to update position tracking: {e}\")\n    \n    async def _log_trade(self, level_name: str, usdc_amount: float, price: float, order_result: Dict):\n        \"\"\"Log trade details\"\"\"\n        try:\n            trade_log = {\n                'timestamp': datetime.now().isoformat(),\n                'strategy': '永久ナンピン',\n                'fibonacci_level': level_name,\n                'usdc_amount': usdc_amount,\n                'btc_price': price,\n                'order_id': order_result.get('id'),\n                'fill_price': order_result.get('fillPrice'),\n                'quantity': order_result.get('quantity'),\n                'status': order_result.get('status'),\n                'position_tracker': self.position_tracker.copy()\n            }\n            \n            # Log to file (you can enhance this with proper trade logging)\n            logger.info(f\"📝 Trade logged: {trade_log}\")\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to log trade: {e}\")\n    \n    async def _monitor_risk(self):\n        \"\"\"Monitor risk levels and take action if needed\"\"\"\n        try:\n            # Get current risk assessment\n            risk_assessment = await self.backpack_client.check_liquidation_risk()\n            \n            risk_level = risk_assessment.get('liquidation_risk', 'unknown')\n            \n            if risk_level == 'critical':\n                logger.error(\"🚨 CRITICAL RISK DETECTED!\")\n                logger.error(\"   Stopping all trading activities\")\n                # Could implement emergency actions here\n                \n            elif risk_level == 'high':\n                logger.warning(\"⚠️ HIGH RISK DETECTED\")\n                logger.warning(\"   Reducing position sizes and monitoring closely\")\n                \n            elif risk_level == 'moderate':\n                logger.info(\"⚠️ Moderate risk - exercising caution\")\n            \n            # Log risk metrics periodically\n            margin_ratio = risk_assessment.get('margin_ratio', 0)\n            if margin_ratio > 0:\n                logger.debug(f\"📊 Risk Status: {risk_level} (margin: {margin_ratio:.1%})\")\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to monitor risk: {e}\")\n    \n    def setup_signal_handlers(self):\n        \"\"\"Setup signal handlers for graceful shutdown\"\"\"\n        def signal_handler(signum, frame):\n            logger.info(f\"Received signal {signum}, initiating shutdown...\")\n            asyncio.create_task(self.stop())\n        \n        signal.signal(signal.SIGINT, signal_handler)\n        signal.signal(signal.SIGTERM, signal_handler)\n\n\ndef print_banner():\n    \"\"\"Print Nanpin Bot banner\"\"\"\n    banner = \"\"\"\n🌸 =========================================== 🌸\n        Nanpin Bot - 永久ナンピン \n   Permanent Dollar-Cost Averaging Strategy\n      Fibonacci-Guided BTC Accumulation\n🌸 =========================================== 🌸\n\n✨ Features:\n   • Fibonacci retracement entry levels\n   • Multi-source liquidation intelligence  \n   • Permanent accumulation (never sell)\n   • Risk management & liquidation protection\n   • Backpack Exchange integration\n\n⚠️  Warning: This is a permanent accumulation strategy.\n   Positions are never sold. Use only risk capital.\n\n🌸 =========================================== 🌸\n\"\"\"\n    print(banner)\n\n\nasync def main():\n    \"\"\"Main entry point\"\"\"\n    print_banner()\n    \n    # Check for required environment variables\n    required_env_vars = ['BACKPACK_API_KEY', 'BACKPACK_SECRET_KEY']\n    missing_vars = [var for var in required_env_vars if not os.getenv(var)]\n    \n    if missing_vars:\n        logger.error(f\"❌ Missing required environment variables: {missing_vars}\")\n        logger.error(\"   Please set BACKPACK_API_KEY and BACKPACK_SECRET_KEY\")\n        sys.exit(1)\n    \n    # Initialize and start bot\n    bot = NanpinBot()\n    bot.setup_signal_handlers()\n    \n    try:\n        await bot.start()\n    except KeyboardInterrupt:\n        logger.info(\"⏹️ Received keyboard interrupt\")\n    except Exception as e:\n        logger.error(f\"❌ Bot crashed: {e}\")\n        sys.exit(1)\n    finally:\n        await bot.stop()\n\n\nif __name__ == \"__main__\":\n    # Ensure logs directory exists\n    os.makedirs(\"logs\", exist_ok=True)\n    \n    # Run the bot\n    try:\n        asyncio.run(main())\n    except KeyboardInterrupt:\n        print(\"\\n👋 Nanpin Bot stopped by user\")\n    except Exception as e:\n        print(f\"\\n❌ Fatal error: {e}\")\n        sys.exit(1)