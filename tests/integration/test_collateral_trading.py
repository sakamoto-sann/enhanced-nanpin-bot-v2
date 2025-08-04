#!/usr/bin/env python3
"""
🎯 Test Trading with Lending Positions as Collateral
Based on Backpack's collateral system where lent assets can be used for trading
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exchanges.backpack_client_fixed import BackpackNanpinClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollateralTradingTester:
    """Test trading using lending positions as collateral"""
    
    def __init__(self):
        self.client = None
    
    async def test_collateral_trading(self):
        """Test if we can trade using collateral from lending positions"""
        try:
            print("🎯 TESTING COLLATERAL-BASED TRADING")
            print("=" * 50)
            print("Testing if lending positions can be used as collateral for trading")
            print()
            
            # Initialize client
            await self._init_client()
            
            # 1. Analyze current collateral status
            print("🏦 STEP 1: Analyze Collateral Status")
            print("-" * 40)
            collateral_info = await self._analyze_collateral()
            
            # 2. Check trading power
            print("\n💪 STEP 2: Calculate Trading Power")
            print("-" * 35)
            trading_power = await self._calculate_trading_power(collateral_info)
            
            # 3. Test small order with collateral
            print("\n🛒 STEP 3: Test Order with Collateral")
            print("-" * 38)
            if trading_power > 5:
                await self._test_collateral_order()
            else:
                print(f"❌ Insufficient trading power: ${trading_power:.2f}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
        finally:
            if self.client:
                await self.client.close()
    
    async def _init_client(self):
        """Initialize client"""
        api_key = os.getenv('BACKPACK_API_KEY')
        secret_key = os.getenv('BACKPACK_SECRET_KEY')
        
        if not api_key or not secret_key:
            raise Exception("Missing API credentials")
        
        self.client = BackpackNanpinClient(api_key, secret_key)
        print("✅ Client initialized")
    
    async def _analyze_collateral(self):
        """Analyze detailed collateral information"""
        try:
            collateral = await self.client.get_collateral_info()
            
            if not collateral:
                print("❌ No collateral data")
                return None
            
            # Extract key metrics
            net_equity = float(collateral.get('netEquity', 0))
            net_equity_available = float(collateral.get('netEquityAvailable', 0))  
            assets_value = float(collateral.get('assetsValue', 0))
            borrow_liability = float(collateral.get('borrowLiability', 0))
            
            print(f"📊 Collateral Analysis:")
            print(f"   Net Equity: ${net_equity:.2f}")
            print(f"   Net Equity Available: ${net_equity_available:.2f}")
            print(f"   Assets Value: ${assets_value:.2f}")
            print(f"   Borrow Liability: ${borrow_liability:.2f}")
            
            # Analyze individual collateral assets
            collateral_assets = collateral.get('collateral', [])
            print(f"\n💰 Collateral Breakdown ({len(collateral_assets)} assets):")
            
            total_collateral_value = 0
            for asset in collateral_assets:
                symbol = asset.get('symbol', 'Unknown')
                lend_qty = float(asset.get('lendQuantity', 0))
                collateral_value = float(asset.get('collateralValue', 0))
                collateral_weight = float(asset.get('collateralWeight', 0))
                mark_price = float(asset.get('assetMarkPrice', 0))
                
                if lend_qty > 0:
                    total_collateral_value += collateral_value
                    print(f"   {symbol}:")
                    print(f"     Lent: {lend_qty:.8f}")
                    print(f"     Mark Price: ${mark_price:.2f}")
                    print(f"     Collateral Value: ${collateral_value:.2f}")
                    print(f"     Collateral Weight: {collateral_weight:.0%}")
            
            print(f"\n✅ Total Collateral Value: ${total_collateral_value:.2f}")
            
            # Check if we have available equity for trading
            if net_equity_available > 0:
                print(f"🎯 Available for Trading: ${net_equity_available:.2f}")
                print("✅ Lending positions ARE being used as collateral!")
            else:
                print("⚠️ No equity currently available for new positions")
            
            return collateral
            
        except Exception as e:
            print(f"❌ Collateral analysis error: {e}")
            return None
    
    async def _calculate_trading_power(self, collateral_info):
        """Calculate available trading power"""
        try:
            if not collateral_info:
                return 0
            
            net_equity_available = float(collateral_info.get('netEquityAvailable', 0))
            
            # In Backpack, net equity available should represent buying power
            # This is the amount that can be used for new positions
            buying_power = net_equity_available
            
            print(f"💪 Trading Power Analysis:")
            print(f"   Net Equity Available: ${net_equity_available:.2f}")
            print(f"   Estimated Buying Power: ${buying_power:.2f}")
            
            if buying_power > 10:
                print("✅ Sufficient power for testing ($5-10 orders)")
            elif buying_power > 5:
                print("⚠️ Limited power - can test small orders only")
            else:
                print("❌ Insufficient power for testing")
            
            return buying_power
            
        except Exception as e:
            print(f"❌ Trading power calculation error: {e}")
            return 0
    
    async def _test_collateral_order(self):
        """Test placing an order using collateral"""
        try:
            print("🛒 Testing order placement with collateral backing...")
            
            # Use a very small test amount
            test_amount = 5.0  # $5 USDC worth
            
            print(f"   Attempting ${test_amount} BTC purchase...")
            print("   This should use your lending collateral if available")
            
            # Try to place the order
            order_result = await self.client.market_buy_btc(
                test_amount,
                reason="Collateral Trading Test"
            )
            
            if order_result:
                print("✅ ORDER EXECUTED SUCCESSFULLY!")
                print(f"   Order ID: {order_result.get('id', 'N/A')}")
                print(f"   Status: {order_result.get('status', 'N/A')}")
                
                if 'quantity' in order_result:
                    btc_qty = float(order_result['quantity'])
                    print(f"   BTC Quantity: {btc_qty:.8f}")
                
                if 'fillPrice' in order_result:
                    fill_price = float(order_result['fillPrice']) 
                    print(f"   Fill Price: ${fill_price:,.2f}")
                
                print("\n🎉 SUCCESS: Your bot CAN place orders using lending collateral!")
                
                # Wait a moment then check position
                await asyncio.sleep(3)
                await self._verify_position_created()
                
                # Offer to close position
                print("\n🔒 Would you like to close this test position? (Safety measure)")
                # For automation, we'll close it automatically
                await self._close_test_position()
                
            else:
                print("❌ Order failed - collateral may not be sufficient")
                
        except Exception as e:
            print(f"❌ Order test error: {e}")
            print(f"   Error details: {type(e).__name__}: {e}")
            
            # Check if it's an insufficient balance error
            if "insufficient" in str(e).lower() or "balance" in str(e).lower():
                print("   This suggests collateral is not available for spot trading")
                print("   You may need to:")
                print("   1. Enable margin trading in your Backpack settings")
                print("   2. Or withdraw some funds from lending")
            else:
                print("   This may be a different API issue")
    
    async def _verify_position_created(self):
        """Verify if position was created"""
        try:
            print("📊 Checking for created position...")
            
            # Check BTC position
            btc_position = await self.client.get_btc_position()
            if btc_position:
                size = float(btc_position.get('size', 0))
                if size > 0:
                    print(f"✅ Position created: {size:.8f} BTC")
                else:
                    print("ℹ️ No BTC position found")
            else:
                print("ℹ️ No positions detected")
                
        except Exception as e:
            print(f"❌ Position check error: {e}")
    
    async def _close_test_position(self):
        """Close the test position for safety"""
        try:
            print("🔒 Attempting to close test position...")
            
            btc_position = await self.client.get_btc_position()
            if not btc_position:
                print("ℹ️ No position to close")
                return
            
            size = float(btc_position.get('size', 0))
            if size <= 0:
                print("ℹ️ No position size to close")
                return
            
            print(f"   Closing {size:.8f} BTC position...")
            
            # Create market sell order
            close_params = {
                'symbol': 'BTC_USDC',
                'side': 'Ask',  # Sell
                'orderType': 'Market',
                'quantity': f"{size:.8f}",
                'timeInForce': 'IOC'
            }
            
            close_result = await self.client._make_request(
                'POST',
                '/api/v1/order',
                close_params,
                signed=True,
                instruction='orderExecute'
            )
            
            if close_result:
                print("✅ Position closing order placed")
                print(f"   Close Order ID: {close_result.get('id', 'N/A')}")
            else:
                print("⚠️ Position closing may have failed")
                
        except Exception as e:
            print(f"❌ Position closing error: {e}")

async def main():
    """Main test execution"""
    print("🎯 BACKPACK COLLATERAL TRADING TEST")
    print("=" * 50)
    print("Testing if lending positions can be used as trading collateral")
    print("Based on official Backpack Exchange documentation")
    print()
    
    tester = CollateralTradingTester()
    await tester.test_collateral_trading()

if __name__ == "__main__":
    asyncio.run(main())