#!/usr/bin/env python3
"""
🎯 Corrected Futures Trading Test
Using correct Backpack API endpoints for futures trading
"""

import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exchanges.backpack_client_fixed import BackpackNanpinClient

async def test_corrected_futures():
    """Test with corrected Backpack futures API endpoints"""
    
    print("🎯 CORRECTED NANPIN FUTURES TEST")
    print("=" * 50)
    print("Using correct Backpack API endpoints")
    print()
    
    client = BackpackNanpinClient(
        os.getenv('BACKPACK_API_KEY'),
        os.getenv('BACKPACK_SECRET_KEY')
    )
    
    try:
        print("1️⃣ TESTING CORRECT POSITION ENDPOINT")
        print("-" * 40)
        
        # Try the correct singular endpoint
        try:
            position = await client._make_request(
                'GET',
                '/api/v1/position',  # SINGULAR, not plural
                signed=True,
                instruction='positionQuery'
            )
            
            if position:
                print("✅ Position endpoint working!")
                print(f"   Response: {position}")
                
                # Check for BTC position
                if isinstance(position, dict):
                    # Single position response
                    if 'BTC' in position.get('symbol', ''):
                        size = float(position.get('size', 0))
                        print(f"   BTC Position: {size:.8f}")
                elif isinstance(position, list):
                    # Multiple positions response
                    btc_pos = None
                    for pos in position:
                        if 'BTC' in pos.get('symbol', ''):
                            btc_pos = pos
                            break
                    
                    if btc_pos:
                        size = float(btc_pos.get('size', 0))
                        print(f"   BTC Position: {size:.8f}")
                    else:
                        print("   No BTC position found")
                        
            else:
                print("   No position data returned")
                
        except Exception as e:
            print(f"   Position endpoint error: {e}")
        
        print("\n2️⃣ TESTING ORDER EXECUTION ENDPOINT")
        print("-" * 40)
        
        # Check if we should use /orders/execute instead of /api/v1/order
        print("   Current bot uses: POST /api/v1/order")
        print("   Documentation suggests: POST /orders/execute")
        print("   Testing current implementation...")
        
        # Test small order placement
        test_amount = 10.0
        safe_amount = await client.calculate_safe_order_size(test_amount)
        
        if safe_amount > 0:
            print(f"   Safe order size: ${safe_amount:.2f}")
            
            # Ask for confirmation
            try:
                confirm = input("\nPlace REAL $10 futures order? Type 'YES': ")
                if confirm != 'YES':
                    print("❌ Order test cancelled")
                    return
            except (EOFError, KeyboardInterrupt):
                print("\n❌ Order test cancelled")
                return
            
            print("\n🚀 PLACING REAL FUTURES ORDER...")
            
            # Use the bot's market_buy_btc method
            order_result = await client.market_buy_btc(
                safe_amount,
                reason="Corrected Futures Test"
            )
            
            if order_result:
                print("✅ ORDER SUCCESSFUL!")
                print(f"   Order ID: {order_result.get('id')}")
                print(f"   Status: {order_result.get('status')}")
                
                if 'quantity' in order_result:
                    btc_qty = float(order_result['quantity'])
                    print(f"   BTC Quantity: {btc_qty:.8f}")
                
                if 'fillPrice' in order_result:
                    fill_price = float(order_result['fillPrice'])
                    print(f"   Fill Price: ${fill_price:,.2f}")
                
                print("\n⏳ Checking position after order...")
                await asyncio.sleep(3)
                
                # Check position with corrected endpoint
                try:
                    updated_position = await client._make_request(
                        'GET',
                        '/api/v1/position',
                        signed=True,
                        instruction='positionQuery'
                    )
                    
                    if updated_position:
                        print("✅ Position updated!")
                        print(f"   New position data: {updated_position}")
                    else:
                        print("   No position data after order")
                        
                except Exception as e:
                    print(f"   Position check error: {e}")
                
                print("\n🎉 FUTURES TRADING: WORKING!")
                print("✅ Your nanpin bot CAN place futures orders")
                print("✅ Bot is ready for live trading")
                
            else:
                print("❌ ORDER FAILED")
                print("   No order result returned")
        else:
            print("❌ Order sizing failed")
    
    except Exception as e:
        print(f"❌ Test error: {e}")
        
        # Provide specific troubleshooting
        error_str = str(e).lower()
        if 'insufficient' in error_str:
            print("\n💡 ISSUE: Funds are in lending, not available for futures")
            print("   SOLUTION: Withdraw some USDC from lending to wallet")
        elif '404' in error_str or 'not found' in error_str:
            print("\n💡 ISSUE: Futures API endpoints or permissions")  
            print("   SOLUTION: Enable futures trading in Backpack account")
        elif 'permission' in error_str:
            print("\n💡 ISSUE: API key lacks futures permissions")
            print("   SOLUTION: Update API key settings in Backpack")
    
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_corrected_futures())