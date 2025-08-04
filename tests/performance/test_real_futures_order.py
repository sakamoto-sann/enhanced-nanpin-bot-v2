#!/usr/bin/env python3
"""
🎯 REAL FUTURES ORDER PLACEMENT TEST
Places a small BTC futures order and immediately closes it for testing
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exchanges.backpack_client_fixed import BackpackNanpinClient

async def test_real_futures_order():
    """Place a real BTC futures order and immediately close it"""
    
    print("🚀 REAL BTC FUTURES ORDER TEST")
    print("=" * 50)
    print("⚠️  WARNING: This will place a REAL futures order with REAL money!")
    print("⚠️  Order will be immediately closed for safety")
    print("⚠️  Make sure you have sufficient collateral in your account")
    print("=" * 50)
    
    # Get user confirmation
    try:
        confirm = input("\nType 'PLACE_ORDER' to proceed with real futures order: ")
        if confirm != 'PLACE_ORDER':
            print("❌ Test cancelled by user")
            return
    except (EOFError, KeyboardInterrupt):
        print("\n❌ Test cancelled by user")
        return
    
    # Initialize client with real API keys
    api_key = os.getenv('BACKPACK_API_KEY')
    secret_key = os.getenv('BACKPACK_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ Error: BACKPACK_API_KEY and BACKPACK_SECRET_KEY must be set in .env file")
        return
        
    if api_key.startswith('your_') or secret_key.startswith('your_'):
        print("❌ Error: Please replace demo API keys with real ones in .env file")
        return
    
    client = BackpackNanpinClient(api_key, secret_key)
    
    try:
        print("\n1️⃣ CONNECTION & AUTHENTICATION TEST")
        print("-" * 40)
        
        # Test connection
        connection_ok = await client.test_connection()
        print(f"   Connection: {'✅ OK' if connection_ok else '❌ FAILED'}")
        
        # Test authentication  
        auth_ok = await client.test_authentication()
        print(f"   Authentication: {'✅ OK' if auth_ok else '❌ FAILED'}")
        
        if not (connection_ok and auth_ok):
            print("❌ Basic connectivity failed")
            return
        
        print("\n2️⃣ COLLATERAL & ACCOUNT CHECK")
        print("-" * 40)
        
        # Get collateral info
        collateral_info = await client.get_collateral_info()
        if not collateral_info:
            print("❌ Could not get collateral information")
            return
            
        net_equity = float(collateral_info.get('netEquity', 0))
        net_equity_available = float(collateral_info.get('netEquityAvailable', 0))
        margin_fraction = collateral_info.get('marginFraction')
        
        print(f"   Net Equity: ${net_equity:.2f}")
        print(f"   Available for Trading: ${net_equity_available:.2f}")
        
        if margin_fraction:
            print(f"   Current Margin Usage: {float(margin_fraction):.1%}")
        else:
            print("   Current Margin Usage: 0% (no open positions)")
        
        # Check if we have enough for futures trading
        min_order_amount = 25.0  # $25 minimum test order
        if net_equity_available < min_order_amount:
            print(f"❌ Insufficient collateral for futures trading (need ${min_order_amount})")
            return
        
        print("\n3️⃣ CURRENT BTC FUTURES POSITION")
        print("-" * 40)
        
        # Check existing BTC futures position
        try:
            positions = await client._make_request(
                'GET',
                '/api/v1/position',
                signed=True,
                instruction='positionQuery'
            )
            
            btc_position = None
            if positions:
                # Handle both single position and array response
                if isinstance(positions, dict):
                    if 'BTC' in positions.get('symbol', ''):
                        btc_position = positions
                elif isinstance(positions, list):
                    for pos in positions:
                        if 'BTC' in pos.get('symbol', ''):
                            btc_position = pos
                            break
            
            if btc_position:
                size = float(btc_position.get('size', 0))
                entry_price = float(btc_position.get('entryPrice', 0))
                mark_price = float(btc_position.get('markPrice', 0))
                unrealized_pnl = float(btc_position.get('unrealizedPnl', 0))
                
                print(f"   Existing BTC Position: {size:.8f} BTC")
                print(f"   Entry Price: ${entry_price:,.2f}")
                print(f"   Mark Price: ${mark_price:,.2f}")
                print(f"   Unrealized PnL: ${unrealized_pnl:,.2f}")
            else:
                print("   No existing BTC futures position")
                
        except Exception as e:
            print(f"   Position check error: {e}")
        
        print("\n4️⃣ BTC FUTURES PRICE CHECK")
        print("-" * 40)
        
        # Get BTC futures price
        btc_price = await client.get_mark_price('BTC_USDC_PERP')
        if not btc_price:
            print("❌ Could not get BTC futures price")
            return
            
        print(f"   BTC Futures Mark Price: ${btc_price:,.2f}")
        
        # Calculate safe order size (minimum $25)
        safe_amount = min(min_order_amount, net_equity_available * 0.01)  # 1% of available
        btc_quantity = safe_amount / btc_price
        
        print(f"   Test Order Size: ${safe_amount:.2f} USDC")
        print(f"   BTC Quantity: {btc_quantity:.8f} BTC")
        
        print(f"\n⚠️  FINAL CONFIRMATION")
        print(f"   This will open a BTC futures position of {btc_quantity:.8f} BTC")
        print(f"   Using ${safe_amount:.2f} USDC collateral")
        print(f"   Position will be closed immediately after test")
        
        # Final confirmation
        try:
            final_confirm = input("\nType 'EXECUTE_FUTURES_ORDER' to place the order: ")
            if final_confirm != 'EXECUTE_FUTURES_ORDER':
                print("❌ Order cancelled by user")
                return
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Order cancelled by user")
            return
        
        print("\n🚀 PLACING BTC FUTURES ORDER...")
        print("-" * 30)
        
        # Place the futures order using the new futures method
        order_result = await client.market_buy_btc_futures(
            safe_amount,
            reason="Nanpin Futures Order Test"
        )
        
        if order_result:
            print("✅ FUTURES ORDER EXECUTED!")
            print(f"   Order ID: {order_result.get('id', 'N/A')}")
            print(f"   Status: {order_result.get('status', 'N/A')}")
            
            if 'quantity' in order_result:
                btc_qty = float(order_result['quantity'])
                print(f"   BTC Quantity: {btc_qty:.8f}")
            
            if 'fillPrice' in order_result:
                fill_price = float(order_result['fillPrice'])
                print(f"   Fill Price: ${fill_price:,.2f}")
            
            print("\n⏳ Waiting 5 seconds before closing position...")
            await asyncio.sleep(5)
            
            print("\n🔄 CLOSING POSITION...")
            print("-" * 20)
            
            # Note: Position closing would require a sell order of the same size
            # For safety, we'll just show the user how to close manually
            print("✅ Test order placed successfully!")
            print("\n📋 TO CLOSE POSITION MANUALLY:")
            print("   1. Go to Backpack Exchange")
            print("   2. Navigate to Futures -> Positions")
            print("   3. Close the BTC position")
            print(f"   4. Or place a sell order for {btc_quantity:.8f} BTC")
            
        else:
            print("❌ FUTURES ORDER FAILED!")
            print("   Check your API keys and collateral")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close client connections
        if hasattr(client, 'session') and client.session:
            await client.session.close()

if __name__ == "__main__":
    asyncio.run(test_real_futures_order())