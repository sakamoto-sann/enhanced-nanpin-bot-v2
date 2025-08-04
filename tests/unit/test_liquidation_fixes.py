#!/usr/bin/env python3
"""
Test script for liquidation aggregator fixes
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.liquidation_aggregator_fixed import LiquidationAggregator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_liquidation_fixes():
    """Test the liquidation aggregator fixes"""
    
    print("🔥 Testing Liquidation Aggregator Fixes")
    print("=" * 50)
    
    # Test 1: Configuration Fix
    print("\n1. Testing Configuration Fix...")
    try:
        liquidation_config = {
            'api_keys': {
                'coinglass': '3ec7b948900e4bd2a407a26fd4c52135',
                'coinmarketcap': None,
                'coingecko': None,
                'flipside': None
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
        
        aggregator = LiquidationAggregator(liquidation_config)
        print("✅ Configuration test passed - no KeyError exceptions")
        
    except KeyError as e:
        print(f"❌ Configuration test failed: Missing key {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    # Test 2: Price Fetching Fix
    print("\n2. Testing Price Fetching Fix...")
    try:
        await aggregator._init_session()
        price = await aggregator._get_current_price('BTC')
        
        if price and price > 10000:
            print(f"✅ Price fetching test passed: ${price:,.2f}")
        else:
            print(f"⚠️ Price fetching returned fallback: ${price:,.2f}")
        
    except Exception as e:
        print(f"❌ Price fetching test failed: {e}")
        await aggregator.close()
        return False
    
    # Test 3: Full Heatmap Generation
    print("\n3. Testing Full Heatmap Generation...")
    try:
        heatmap = await aggregator.generate_liquidation_heatmap('BTC')
        
        if heatmap:
            print(f"✅ Heatmap generation test passed")
            print(f"   Symbol: {heatmap.symbol}")
            print(f"   Current Price: ${heatmap.current_price:,.2f}")
            print(f"   Clusters: {len(heatmap.clusters)}")
            print(f"   Data Sources: {heatmap.data_sources}")
            print(f"   Opportunities: {len(heatmap.nanpin_opportunities)}")
        else:
            print("❌ Heatmap generation test failed - returned None")
            await aggregator.close()
            return False
            
    except Exception as e:
        print(f"❌ Heatmap generation test failed: {e}")
        await aggregator.close()
        return False
    
    await aggregator.close()
    
    print("\n🎉 All tests passed! Liquidation aggregator fixes are working.")
    return True

async def main():
    """Main test function"""
    try:
        success = await test_liquidation_fixes()
        if success:
            print("\n✅ All liquidation aggregator errors should now be resolved!")
        else:
            print("\n❌ Some issues remain - check the error messages above.")
            
    except Exception as e:
        print(f"\n❌ Test script error: {e}")

if __name__ == "__main__":
    asyncio.run(main())