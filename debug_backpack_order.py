#!/usr/bin/env python3
"""
🔍 Debug Backpack Order Requirements
Test exact order parameters and requirements
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exchanges.backpack_client_fixed import BackpackNanpinClient

async def debug_order_requirements():
    """Debug what Backpack actually needs for orders"""
    
    print("🔍 DEBUGGING BACKPACK ORDER REQUIREMENTS")
    print("=" * 50)
    
    client = BackpackNanpinClient(
        os.getenv('BACKPACK_API_KEY'),
        os.getenv('BACKPACK_SECRET_KEY')
    )
    
    try:
        # 1. Check market info for BTC_USDC
        print("1️⃣ MARKET INFORMATION")
        print("-" * 30)
        
        try:
            # Get ticker info which might have market details
            ticker = await client._make_request(
                'GET',
                '/api/v1/ticker',
                {'symbol': 'BTC_USDC'},
                signed=False
            )
            
            if ticker:
                print(f"   BTC_USDC Ticker: {ticker}")
            else:
                print("   No ticker data")
        except Exception as e:
            print(f"   Ticker error: {e}")
        
        # 2. Check if we need to enable futures trading settings
        print("\n2️⃣ ACCOUNT SETTINGS")
        print("-" * 30)
        
        try:
            # Try to get account settings
            settings = await client._make_request(
                'GET',
                '/api/v1/capital',
                signed=True,
                instruction='balanceQuery'
            )
            
            if settings:
                print(f"   Account info: {settings}")
        except Exception as e:
            print(f"   Settings error: {e}")
        
        # 3. Test order with different parameters
        print("\n3️⃣ ORDER PARAMETER TESTING")
        print("-" * 30)
        
        btc_price = await client.get_btc_price()
        print(f"   Current BTC Price: ${btc_price:,.2f}")
        
        # Test different order formats
        test_cases = [
            {
                'name': 'Current bot format',
                'params': {
                    'symbol': 'BTC_USDC',
                    'side': 'Bid',
                    'orderType': 'Market',
                    'quoteQuantity': '10.00',
                    'timeInForce': 'IOC'
                }
            },
            {
                'name': 'With quantity instead of quoteQuantity',
                'params': {
                    'symbol': 'BTC_USDC',
                    'side': 'Bid',
                    'orderType': 'Market',
                    'quantity': f"{10.0 / btc_price:.8f}",
                    'timeInForce': 'IOC'
                }
            },
            {
                'name': 'Limit order format',
                'params': {
                    'symbol': 'BTC_USDC',
                    'side': 'Bid',
                    'orderType': 'Limit',
                    'quantity': f"{10.0 / btc_price:.8f}",
                    'price': f"{btc_price:.2f}",
                    'timeInForce': 'GTC'
                }
            }
        ]
        
        for test in test_cases:
            print(f"\n   🧪 Testing: {test['name']}")
            print(f"      Parameters: {test['params']}")
            
            try:
                # Don't actually place the order, just test the API call structure
                print("      (Test mode - not placing real order)")
                
                # We can test by seeing what error we get
                # The "insufficient funds" vs other errors tells us if params are right
                
            except Exception as e:
                print(f"      Error: {e}")
        
        # 4. Check if account needs margin trading enabled
        print("\n4️⃣ MARGIN/FUTURES SETTINGS CHECK")
        print("-" * 30)
        
        # Check if we can access futures-specific endpoints
        try:
            positions = await client._make_request(
                'GET',
                '/api/v1/position',
                signed=True,
                instruction='positionQuery'
            )
            
            if positions is not None:  # Even empty list means endpoint works
                print("   ✅ Futures API access: WORKING")
                print(f"   Current positions: {positions if positions else 'None'}")
            else:
                print("   ❌ Futures API access: FAILED")
                
        except Exception as e:
            print(f"   ❌ Futures API error: {e}")
        
        # 5. Final recommendation
        print("\n5️⃣ ANALYSIS & RECOMMENDATIONS")
        print("-" * 30)
        
        collateral = await client.get_collateral_info()
        if collateral:
            net_equity = float(collateral.get('netEquityAvailable', 0))
            print(f"   Available Collateral: ${net_equity:.2f}")
            
            if net_equity > 10:
                print("   💡 LIKELY ISSUES:")
                print("      1. Account may need futures trading enabled")
                print("      2. API key may lack futures permissions")
                print("      3. Market order format may be incorrect")
                print("      4. Minimum order size requirements")
                print()
                print("   🔧 SOLUTIONS TO TRY:")
                print("      1. Enable futures/margin trading in Backpack settings")
                print("      2. Check API key permissions")
                print("      3. Try limit orders instead of market orders")
                print("      4. Use quantity parameter instead of quoteQuantity")
    
    except Exception as e:
        print(f"❌ Debug error: {e}")
    
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(debug_order_requirements())