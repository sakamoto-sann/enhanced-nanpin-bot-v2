#!/usr/bin/env python3
"""
🚀 FUTURES TRADING TEST
Test futures trading with collateral - should work without spot USDC
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exchanges.backpack_client_fixed import BackpackNanpinClient

async def test_futures_trading():
    """Test futures trading with collateral"""
    
    print("🚀 FUTURES TRADING WITH COLLATERAL TEST")
    print("=" * 50)
    
    client = BackpackNanpinClient(
        os.getenv('BACKPACK_API_KEY'),
        os.getenv('BACKPACK_SECRET_KEY')
    )
    
    try:
        # 1. Check if BTC_USDC is spot or futures
        print("1️⃣ ANALYZING BTC_USDC CONTRACT TYPE")
        print("-" * 30)
        
        # Get all symbols to see the pattern
        try:
            symbols = await client._make_request(
                'GET',
                '/api/v1/ticker',
                signed=False
            )
            
            if symbols:
                btc_symbols = [s['symbol'] for s in symbols if 'BTC' in s['symbol']]
                print(f"All BTC symbols: {btc_symbols}")
                
                # Look for perpetual futures patterns
                perp_symbols = [s['symbol'] for s in symbols if 'PERP' in s['symbol'] or 'USD' in s['symbol']]
                print(f"Perpetual/USD symbols: {perp_symbols[:10]}")
                
        except Exception as e:
            print(f"Symbol analysis error: {e}")
        
        # 2. Check positions to confirm futures access
        print("\n2️⃣ CHECKING FUTURES POSITIONS")
        print("-" * 30)
        
        positions = await client._make_request(
            'GET',
            '/api/v1/position',
            signed=True,
            instruction='positionQuery'
        )
        
        if positions is not None:
            print(f"✅ Futures positions endpoint working: {positions}")
            if len(positions) == 0:
                print("   No current positions (normal for new account)")
        
        # 3. Test if BTC_USDC behaves like futures or spot
        print("\n3️⃣ TESTING BTC_USDC ORDER BEHAVIOR")
        print("-" * 30)
        
        btc_price = await client.get_btc_price()
        collateral = await client.get_collateral_info()
        net_equity = float(collateral.get('netEquityAvailable', 0)) if collateral else 0
        
        print(f"BTC Price: ${btc_price:,.2f}")
        print(f"Available Collateral: ${net_equity:.2f}")
        
        # Test very small futures order
        small_btc_amount = 0.0001  # $11-12 worth
        required_collateral = small_btc_amount * btc_price
        
        print(f"Testing {small_btc_amount} BTC (${required_collateral:.2f} worth)")
        
        if net_equity > required_collateral * 3:  # 3x margin safety
            print("✅ More than enough collateral for test order")
            
            order_params = {
                'symbol': 'BTC_USDC',
                'side': 'Bid',
                'orderType': 'Market',
                'quantity': f'{small_btc_amount:.4f}',
                'timeInForce': 'IOC'
            }
            
            print(f"Order params: {order_params}")
            
            try:
                result = await client._make_request(
                    'POST',
                    '/api/v1/order',
                    order_params,
                    signed=True,
                    instruction='orderExecute'
                )
                
                print(f"✅ SUCCESS: Futures order placed: {result}")
                
                # Immediately close the position if it opened
                if result and 'orderId' in result:
                    print("🛑 Closing position immediately for safety...")
                    # Close with opposite order
                    close_params = {
                        'symbol': 'BTC_USDC',
                        'side': 'Ask',  # Opposite side to close
                        'orderType': 'Market',
                        'quantity': f'{small_btc_amount:.4f}',
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
                        print(f"✅ Position closed: {close_result}")
                    except Exception as e:
                        print(f"Close position error: {e}")
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"❌ Order failed: {e}")
                
                if 'insufficient' in error_msg:
                    print("\n💡 DIAGNOSIS:")
                    print("   - Collateral exists but not recognized for this symbol")
                    print("   - BTC_USDC might be SPOT trading, not futures")
                    print("   - Or account needs futures trading enabled")
                elif 'invalid symbol' in error_msg:
                    print("   - BTC_USDC not valid for futures trading")
                elif 'quantity' in error_msg:
                    print("   - Quantity precision issue")
        
        # 4. Test with a known futures symbol if BTC_USDC fails
        print("\n4️⃣ TESTING ALTERNATIVE FUTURES SYMBOLS")
        print("-" * 30)
        
        # Common futures symbol patterns for BTC
        alternative_symbols = [
            'BTCUSD-PERP',
            'BTC-PERP', 
            'BTCUSDC-PERP',
            'SOL_USDC',  # Try SOL since we have SOL in lending
            'ETH_USDC'   # Try ETH since we have ETH in lending
        ]
        
        for symbol in alternative_symbols:
            print(f"\n   Testing {symbol}:")
            
            test_params = {
                'symbol': symbol,
                'side': 'Bid',
                'orderType': 'Market', 
                'quantity': '0.001',
                'timeInForce': 'IOC'
            }
            
            try:
                result = await client._make_request(
                    'POST',
                    '/api/v1/order',
                    test_params,
                    signed=True,
                    instruction='orderExecute'
                )
                
                print(f"   ✅ {symbol} works: {result}")
                
            except Exception as e:
                error_msg = str(e).lower()
                if 'invalid symbol' in error_msg or 'unknown symbol' in error_msg:
                    print(f"   ❌ {symbol}: Not available")
                elif 'insufficient' in error_msg:
                    print(f"   💡 {symbol}: Valid symbol, collateral issue")
                else:
                    print(f"   ❓ {symbol}: {e}")
        
        # 5. Account configuration check
        print("\n5️⃣ ACCOUNT CONFIGURATION ANALYSIS")
        print("-" * 30)
        
        print("Checking if account is configured for futures trading...")
        
        # Check if we can access futures-specific endpoints
        futures_endpoints = [
            '/api/v1/position',  # Already tested above
            '/api/v1/leverage',
            '/api/v1/margin'
        ]
        
        for endpoint in futures_endpoints:
            try:
                result = await client._make_request(
                    'GET',
                    endpoint,
                    signed=True,
                    instruction='balanceQuery'
                )
                print(f"   ✅ {endpoint}: Working")
            except Exception as e:
                if '404' in str(e):
                    print(f"   ❌ {endpoint}: Not found")
                else:
                    print(f"   ❓ {endpoint}: {e}")
        
        # Final recommendation
        print("\n6️⃣ FUTURES TRADING CONCLUSION")
        print("-" * 30)
        print("ANALYSIS RESULTS:")
        
        if net_equity > 20:
            print(f"✅ Sufficient collateral: ${net_equity:.2f}")
            print("❓ Issue is likely:")
            print("   1. BTC_USDC is SPOT trading (needs USDC in spot wallet)")
            print("   2. OR account needs futures trading enabled in settings") 
            print("   3. OR need to use correct futures symbol (BTC-PERP)")
            print("   4. OR API key lacks futures trading permissions")
            print("\n🔧 SOLUTIONS TO TRY:")
            print("   1. Enable futures trading in Backpack account settings")
            print("   2. Check API key permissions include futures")
            print("   3. Try correct futures symbol if BTC_USDC is spot")
        else:
            print("❌ Insufficient collateral for futures trading")
        
    except Exception as e:
        print(f"❌ Futures test error: {e}")
    
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_futures_trading())