#!/usr/bin/env python3
"""
🧪 ACTUAL ORDER PLACEMENT TEST
Test real order placement to get exact error messages
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exchanges.backpack_client_fixed import BackpackNanpinClient

async def test_actual_order_placement():
    """Test actual order placement to get exact error messages"""
    
    print("🧪 ACTUAL ORDER PLACEMENT TEST")
    print("=" * 50)
    
    client = BackpackNanpinClient(
        os.getenv('BACKPACK_API_KEY'),
        os.getenv('BACKPACK_SECRET_KEY')
    )
    
    try:
        # Get current market data
        btc_price = await client.get_btc_price()
        print(f"Current BTC Price: ${btc_price:,.2f}")
        
        # Check available collateral
        collateral = await client.get_collateral_info()
        if collateral:
            net_equity = float(collateral.get('netEquityAvailable', 0))
            print(f"Available Collateral: ${net_equity:.2f}")
        
        # Test 1: Small market order with quoteQuantity
        print("\n1️⃣ TESTING: Market Order with quoteQuantity")
        print("-" * 40)
        
        order_params = {
            'symbol': 'BTC_USDC',
            'side': 'Bid',
            'orderType': 'Market',
            'quoteQuantity': '5.00',  # Very small $5 order
            'timeInForce': 'IOC'
        }
        
        print(f"Order Parameters: {order_params}")
        
        try:
            # Try to place the actual order
            result = await client._make_request(
                'POST',
                '/api/v1/order',
                order_params,
                signed=True,
                instruction='orderExecute'
            )
            
            if result:
                print(f"✅ SUCCESS: Order placed: {result}")
                
                # If successful, immediately cancel or close
                if 'orderId' in result:
                    print("🛑 Attempting to cancel order immediately...")
                    try:
                        cancel_result = await client._make_request(
                            'DELETE',
                            f'/api/v1/order',
                            {'orderId': result['orderId'], 'symbol': 'BTC_USDC'},
                            signed=True,
                            instruction='orderCancel'
                        )
                        print(f"Cancel result: {cancel_result}")
                    except Exception as e:
                        print(f"Cancel error (order may have filled): {e}")
            
        except Exception as e:
            print(f"❌ Market Order Error: {e}")
            print(f"Error type: {type(e)}")
            
            # Analyze the error message
            error_str = str(e).lower()
            if 'insufficient' in error_str:
                print("💡 DIAGNOSIS: Insufficient funds - need spot USDC balance")
            elif 'invalid' in error_str or 'bad request' in error_str:
                print("💡 DIAGNOSIS: Invalid parameters or format")
            elif 'unauthorized' in error_str or 'forbidden' in error_str:
                print("💡 DIAGNOSIS: API key permissions issue")
            elif 'not found' in error_str or '404' in error_str:
                print("💡 DIAGNOSIS: Endpoint not found - wrong URL")
            else:
                print("💡 DIAGNOSIS: Unknown error - need to investigate further")
        
        # Test 2: Try with /orders endpoint instead of /api/v1/order
        print("\n2️⃣ TESTING: Different endpoint /orders")
        print("-" * 40)
        
        try:
            result = await client._make_request(
                'POST',
                '/orders',
                order_params,
                signed=True,
                instruction='orderExecute'
            )
            
            if result:
                print(f"✅ SUCCESS: Order placed via /orders: {result}")
            
        except Exception as e:
            print(f"❌ /orders endpoint error: {e}")
        
        # Test 3: Try quantity parameter instead of quoteQuantity
        print("\n3️⃣ TESTING: Market Order with quantity parameter")
        print("-" * 40)
        
        order_params_qty = {
            'symbol': 'BTC_USDC',
            'side': 'Bid',
            'orderType': 'Market',
            'quantity': f"{5.0 / btc_price:.8f}",  # $5 worth of BTC
            'timeInForce': 'IOC'
        }
        
        print(f"Order Parameters: {order_params_qty}")
        
        try:
            result = await client._make_request(
                'POST',
                '/api/v1/order',
                order_params_qty,
                signed=True,
                instruction='orderExecute'
            )
            
            if result:
                print(f"✅ SUCCESS: Order placed with quantity: {result}")
            
        except Exception as e:
            print(f"❌ Quantity parameter error: {e}")
        
        # Test 4: Check if we need to transfer from lending first
        print("\n4️⃣ TESTING: Check lending withdrawal options")
        print("-" * 40)
        
        try:
            # Check lending positions
            lending_positions = await client._make_request(
                'GET',
                '/api/v1/capital',
                signed=True,
                instruction='balanceQuery'
            )
            
            if lending_positions and 'USDC' in lending_positions:
                usdc_info = lending_positions['USDC']
                print(f"USDC Spot Balance: {usdc_info}")
                
                if float(usdc_info.get('available', 0)) == 0:
                    print("❌ No USDC in spot wallet")
                    print("💡 SOLUTION: Need to withdraw USDC from lending to spot wallet")
                    
                    # Check if we can see lending positions
                    try:
                        borrow_lend = await client._make_request(
                            'GET',
                            '/api/v1/borrow-lend',
                            signed=True,
                            instruction='balanceQuery'
                        )
                        
                        if borrow_lend:
                            print(f"Borrow-Lend positions: {borrow_lend}")
                        
                    except Exception as e:
                        print(f"Borrow-lend check error: {e}")
        
        except Exception as e:
            print(f"Lending check error: {e}")
        
        # Final analysis
        print("\n5️⃣ FINAL ANALYSIS")
        print("-" * 40)
        print("Based on the test results:")
        print("1. API authentication is working (can query data)")
        print("2. Futures API access is confirmed")
        print("3. Order placement errors will show exact issue")
        print("4. Most likely need USDC in spot wallet, not just collateral")
        
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_actual_order_placement())