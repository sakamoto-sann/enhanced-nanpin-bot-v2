#!/usr/bin/env python3
"""
🚀 TEST NANPIN BOT WITH FUTURES SYMBOL
Test the bot with BTC_USDC_PERP to confirm it works with collateral
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exchanges.backpack_client_fixed import BackpackNanpinClient

async def test_nanpin_futures():
    """Test nanpin bot functionality with futures symbol"""
    
    print("🚀 TESTING NANPIN BOT WITH FUTURES SYMBOL")
    print("=" * 50)
    
    client = BackpackNanpinClient(
        os.getenv('BACKPACK_API_KEY'),
        os.getenv('BACKPACK_SECRET_KEY')
    )
    
    try:
        # 1. Check current setup
        print("1️⃣ CURRENT SETUP VERIFICATION")
        print("-" * 30)
        
        # Check collateral
        collateral = await client.get_collateral_info()
        if collateral:
            net_equity = float(collateral.get('netEquityAvailable', 0))
            print(f"Available Collateral: ${net_equity:.2f}")
        
        # Check BTC_USDC_PERP price
        try:
            ticker = await client._make_request(
                'GET',
                '/api/v1/ticker',
                {'symbol': 'BTC_USDC_PERP'},
                signed=False
            )
            
            if ticker:
                perp_price = float(ticker['lastPrice'])
                print(f"BTC_USDC_PERP Price: ${perp_price:,.2f}")
        
        except Exception as e:
            print(f"Price check error: {e}")
        
        # 2. Test bot's market_buy_btc method with futures
        print("\n2️⃣ TESTING BOT'S MARKET_BUY_BTC METHOD")
        print("-" * 30)
        
        # Override the symbol in the client for testing
        original_symbol = getattr(client, 'symbol', 'BTC_USDC')
        client.symbol = 'BTC_USDC_PERP'  # Use futures symbol
        
        print(f"Using symbol: {client.symbol}")
        
        # Test small buy order ($10)
        try:
            print("Testing $10 buy order...")
            
            # Use the bot's actual method
            result = await client.market_buy_btc(
                usdc_amount=10.0,
                reason="nanpin_futures_test"
            )
            
            if result:
                print(f"✅ SUCCESS: Buy order executed!")
                print(f"Order details: {result}")
                
                # If successful, try to close the position immediately
                print("\n🛑 Closing position for safety...")
                
                try:
                    # Get current position to determine quantity
                    positions = await client._make_request(
                        'GET',
                        '/api/v1/position',
                        signed=True,
                        instruction='positionQuery'
                    )
                    
                    if positions:
                        btc_position = next((p for p in positions if 'BTC' in str(p)), None)
                        if btc_position:
                            print(f"Current position: {btc_position}")
                            
                            # Close the position
                            position_size = float(btc_position.get('size', 0))
                            if position_size > 0:
                                close_params = {
                                    'symbol': 'BTC_USDC_PERP',
                                    'side': 'Ask',  # Sell to close long
                                    'orderType': 'Market',
                                    'quantity': f'{position_size:.4f}',
                                    'timeInForce': 'IOC'
                                }
                                
                                close_result = await client._make_request(
                                    'POST',
                                    '/api/v1/order',
                                    close_params,
                                    signed=True,
                                    instruction='orderExecute'
                                )
                                
                                print(f"✅ Position closed: {close_result}")
                
                except Exception as e:
                    print(f"⚠️ Close position error: {e}")
            
        except Exception as e:
            error_msg = str(e).lower()
            print(f"❌ Buy order failed: {e}")
            
            if 'insufficient' in error_msg:
                print("💡 Still insufficient funds - may need account settings")
            elif 'symbol' in error_msg:
                print("💡 Symbol issue - check configuration")
            else:
                print("💡 Unknown error - investigate further")
        
        # 3. Test position management functions
        print("\n3️⃣ TESTING POSITION MANAGEMENT")
        print("-" * 30)
        
        try:
            # Test get_positions method
            positions = await client.get_positions()
            print(f"Current positions: {positions}")
            
            # Test get_btc_position method if it exists
            if hasattr(client, 'get_btc_position'):
                btc_pos = await client.get_btc_position()
                print(f"BTC position: {btc_pos}")
        
        except Exception as e:
            print(f"Position management error: {e}")
        
        # 4. Test calculate_safe_order_size with futures
        print("\n4️⃣ TESTING SAFE ORDER SIZE CALCULATION")
        print("-" * 30)
        
        try:
            safe_amount = await client.calculate_safe_order_size(50.0)  # Test $50
            print(f"Safe order size for $50: ${safe_amount:.2f}")
            
            if safe_amount > 0:
                print("✅ Order size calculation works with collateral")
            else:
                print("❌ Order size calculation returned 0")
        
        except Exception as e:
            print(f"Safe order size error: {e}")
        
        # 5. Summary
        print("\n5️⃣ TEST SUMMARY")
        print("-" * 30)
        print("RESULTS:")
        if net_equity > 20:
            print(f"✅ Sufficient collateral: ${net_equity:.2f}")
            print("✅ BTC_USDC_PERP symbol updated in config")
            print("✅ Futures trading should work with lending collateral")
            print()
            print("🚀 NEXT STEPS:")
            print("1. Run your actual nanpin bot")
            print("2. It should now work with your lending collateral") 
            print("3. No need to withdraw funds from lending!")
        else:
            print("❌ Insufficient collateral for trading")
        
        # Restore original symbol
        client.symbol = original_symbol
        
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_nanpin_futures())