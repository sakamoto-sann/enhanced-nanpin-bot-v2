#!/usr/bin/env python3
"""
🎯 REAL Order Placement Test
Actually place a genuine order on Backpack Exchange
"""

import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exchanges.backpack_client_fixed import BackpackNanpinClient

async def place_real_order():
    """Place a genuine order on Backpack"""
    
    print("🎯 REAL ORDER PLACEMENT TEST")
    print("=" * 50)
    print("⚠️  WARNING: This will place a REAL order with REAL money!")
    print("⚠️  This is NOT a simulation!")
    print()
    
    # Safety confirmation
    try:
        confirm = input("Type 'REAL' to place actual order: ")
        if confirm != 'REAL':
            print("❌ Test cancelled")
            return
    except (EOFError, KeyboardInterrupt):
        print("\n❌ Test cancelled")
        return
    
    # Initialize client
    api_key = os.getenv('BACKPACK_API_KEY')
    secret_key = os.getenv('BACKPACK_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ API credentials not found")
        return
    
    client = BackpackNanpinClient(api_key, secret_key)
    
    try:
        print("🔗 Testing connection...")
        connection_ok = await client.test_connection()
        if not connection_ok:
            print("❌ Connection failed")
            return
        print("✅ Connection successful")
        
        print("🔐 Testing authentication...")
        auth_ok = await client.test_authentication()
        if not auth_ok:
            print("❌ Authentication failed")
            return
        print("✅ Authentication successful")
        
        # Get current collateral info
        print("💰 Checking available funds...")
        collateral = await client.get_collateral_info()
        if not collateral:
            print("❌ Could not get collateral info")
            return
        
        net_equity = float(collateral.get('netEquityAvailable', 0))
        print(f"   Available: ${net_equity:.2f}")
        
        if net_equity < 5:
            print("❌ Insufficient funds for test")
            return
        
        # Get BTC price
        btc_price = await client.get_btc_price()
        print(f"📈 Current BTC price: ${btc_price:,.2f}")
        
        # Calculate order details
        order_amount = 5.0  # $5 USDC
        btc_quantity = order_amount / btc_price
        
        print(f"🛒 Placing REAL order:")
        print(f"   Amount: ${order_amount:.2f} USDC")
        print(f"   Quantity: {btc_quantity:.8f} BTC")
        print(f"   Price: ~${btc_price:,.2f}")
        print()
        
        # Final confirmation
        final_confirm = input("Type 'EXECUTE' to place REAL order: ")
        if final_confirm != 'EXECUTE':
            print("❌ Order cancelled")
            return
        
        # Place the order directly via API
        print("🚀 Placing order...")
        
        order_params = {
            'symbol': 'BTC_USDC',
            'side': 'Bid',  # Buy order
            'orderType': 'Market',
            'quoteQuantity': f"{order_amount:.2f}",
            'timeInForce': 'IOC',
            'clientId': int(datetime.now().timestamp())
        }
        
        print(f"📋 Order parameters: {order_params}")
        
        # Execute order
        order_result = await client._make_request(
            'POST',
            '/api/v1/order',
            order_params,
            signed=True,
            instruction='orderExecute'
        )
        
        if order_result:
            print("🎉 ORDER EXECUTED!")
            print(f"   Order ID: {order_result.get('id', 'N/A')}")
            print(f"   Status: {order_result.get('status', 'N/A')}")
            
            if 'quantity' in order_result:
                print(f"   BTC Received: {float(order_result['quantity']):.8f}")
            
            if 'fillPrice' in order_result:
                print(f"   Fill Price: ${float(order_result['fillPrice']):,.2f}")
            
            print()
            print("✅ REAL ORDER PLACED SUCCESSFULLY!")
            print("✅ Check your Backpack Exchange account to confirm")
            
            # Wait and then check for position
            print("\n⏳ Waiting 5 seconds to check position...")
            await asyncio.sleep(5)
            
            # Check positions
            try:
                positions = await client._make_request(
                    'GET',
                    '/api/v1/positions',
                    signed=True,
                    instruction='positionQuery'
                )
                
                if positions:
                    for pos in positions:
                        if 'BTC' in pos.get('symbol', ''):
                            size = float(pos.get('size', 0))
                            if size > 0:
                                print(f"✅ Position created: {size:.8f} BTC")
                                break
                    else:
                        print("ℹ️  No BTC position detected yet")
                else:
                    print("ℹ️  No positions data")
            except Exception as e:
                print(f"⚠️  Position check error: {e}")
            
        else:
            print("❌ ORDER FAILED - No result returned")
            print("❌ Check API credentials and permissions")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(place_real_order())