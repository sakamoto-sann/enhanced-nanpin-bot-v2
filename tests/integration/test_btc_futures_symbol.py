#!/usr/bin/env python3
"""
🚀 TEST CORRECT BTC FUTURES SYMBOL
Test BTC_USDC_PERP and other futures symbols with collateral
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exchanges.backpack_client_fixed import BackpackNanpinClient

async def test_btc_futures():
    """Test correct BTC futures symbol"""
    
    print("🚀 TESTING CORRECT BTC FUTURES SYMBOLS")
    print("=" * 50)
    
    client = BackpackNanpinClient(
        os.getenv('BACKPACK_API_KEY'),
        os.getenv('BACKPACK_SECRET_KEY')
    )
    
    try:
        # Get collateral info
        collateral = await client.get_collateral_info()
        if collateral:
            net_equity = float(collateral.get('netEquityAvailable', 0))
            print(f"Available Collateral: ${net_equity:.2f}")
        
        btc_price = await client.get_btc_price()
        print(f"Current BTC Price: ${btc_price:,.2f}")
        
        # Test possible BTC futures symbols
        futures_symbols = [
            'BTC_USDC_PERP',
            'BTCUSDC_PERP', 
            'BTC-USDC-PERP',
            'BTCUSD_PERP',
            'BTC_USD_PERP',
            'SOL_USDC_PERP',  # Test SOL since docs mention it
            'ETH_USDC_PERP'   # Test ETH as alternative
        ]
        
        print(f"\n1️⃣ TESTING FUTURES SYMBOLS")
        print("-" * 30)
        
        for symbol in futures_symbols:
            print(f"\n🧪 Testing {symbol}:")
            
            # First check if symbol exists by getting ticker
            try:
                ticker = await client._make_request(
                    'GET',
                    '/api/v1/ticker',
                    {'symbol': symbol},
                    signed=False
                )
                
                if ticker:
                    print(f"   ✅ Symbol exists: Price ${float(ticker['lastPrice']):,.2f}")
                    
                    # Now test a small order
                    order_params = {
                        'symbol': symbol,
                        'side': 'Bid',
                        'orderType': 'Market',
                        'quantity': '0.001',  # Small amount
                        'timeInForce': 'IOC'
                    }
                    
                    print(f"   📝 Testing order: {order_params}")
                    
                    try:
                        result = await client._make_request(
                            'POST',
                            '/api/v1/order',
                            order_params,
                            signed=True,
                            instruction='orderExecute'
                        )
                        
                        print(f"   🎉 SUCCESS! Order placed: {result}")
                        
                        # If successful, immediately close the position
                        if result and 'orderId' in result:
                            print("   🛑 Closing position immediately...")
                            
                            close_params = {
                                'symbol': symbol,
                                'side': 'Ask',  # Opposite side
                                'orderType': 'Market', 
                                'quantity': '0.001',
                                'timeInForce': 'IOC'
                            }
                            
                            try:
                                close_result = await client._make_request(
                                    'POST',
                                    '/api/v1/order',
                                    close_params,
                                    signed=True,
                                    instruction='orderExecute'
                                )
                                print(f"   ✅ Position closed: {close_result}")
                            except Exception as e:
                                print(f"   ⚠️ Close error: {e}")
                        
                        # This is the correct symbol!
                        print(f"\n🎯 FOUND WORKING FUTURES SYMBOL: {symbol}")
                        break
                        
                    except Exception as e:
                        error_msg = str(e).lower()
                        if 'insufficient' in error_msg:
                            print(f"   💡 Symbol works but needs more collateral")
                        elif 'quantity' in error_msg:
                            print(f"   ⚠️ Symbol works but quantity precision issue")
                        else:
                            print(f"   ❌ Order error: {e}")
                
                else:
                    print(f"   ❌ Symbol not found")
                    
            except Exception as e:
                error_msg = str(e).lower()
                if 'symbol' in error_msg or 'not found' in error_msg:
                    print(f"   ❌ Symbol doesn't exist")
                else:
                    print(f"   ❓ Unknown error: {e}")
        
        # If no futures symbols work, provide solution
        print(f"\n2️⃣ SOLUTION BASED ON FINDINGS")
        print("-" * 30)
        
        print("ANALYSIS:")
        print("1. BTC_USDC is confirmed as SPOT trading pair")
        print("2. Futures symbols need to be tested")
        print("3. You have two options:")
        print()
        print("📋 OPTION A: Use Futures Symbol (Recommended)")
        print("   - Update bot to use correct futures symbol (e.g., BTC_USDC_PERP)")
        print("   - Will use your $75+ collateral directly")
        print("   - No need to move funds from lending")
        print()
        print("📋 OPTION B: Use Current Spot Symbol")
        print("   - Withdraw $20-30 USDC from lending to spot wallet")
        print("   - Keep current BTC_USDC symbol")
        print("   - Bot will work immediately after funding")
        
        # Show how to fix the bot
        print(f"\n3️⃣ HOW TO FIX YOUR BOT")
        print("-" * 30)
        print("If a futures symbol works above, update your bot config:")
        print()
        print("In your bot configuration, change:")
        print("   FROM: symbol = 'BTC_USDC'")
        print("   TO:   symbol = 'BTC_USDC_PERP' (or whatever worked)")
        print()
        print("This will enable futures trading with your lending collateral!")
        
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_btc_futures())