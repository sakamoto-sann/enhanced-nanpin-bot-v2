#!/usr/bin/env python3
"""
🎯 IMMEDIATE REAL FUTURES ORDER EXECUTION
Executes BTC futures order immediately without prompts as requested
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exchanges.backpack_client_fixed import BackpackNanpinClient

async def execute_real_futures_order():
    """Execute a real BTC futures order immediately"""
    
    print("🚀 EXECUTING REAL BTC FUTURES ORDER NOW")
    print("=" * 50)
    print("⚡ IMMEDIATE EXECUTION - NO PROMPTS")
    print("=" * 50)
    
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
        
        # Check if we have enough for futures trading - use minimum required
        min_order_amount = 5.0  # $5 minimum per Backpack requirements
        test_order_amount = 10.0  # $10 test order to be safe
        if net_equity_available < test_order_amount:
            print(f"❌ Insufficient collateral for futures trading (need ${test_order_amount})")
            return
        
        print("\n3️⃣ BTC FUTURES PRICE CHECK")
        print("-" * 40)
        
        # Get BTC futures price
        btc_price = await client.get_mark_price('BTC_USDC_PERP')
        if not btc_price:
            print("❌ Could not get BTC futures price")
            return
            
        print(f"   BTC Futures Mark Price: ${btc_price:,.2f}")
        
        # Calculate safe order size (minimum $5, using $10 for test)
        safe_amount = test_order_amount  # Use fixed $10 for test
        btc_quantity = safe_amount / btc_price
        
        print(f"   Order Size: ${safe_amount:.2f} USDC")
        print(f"   BTC Quantity: {btc_quantity:.8f} BTC")
        
        print(f"\n🚀 PLACING BTC FUTURES ORDER NOW...")
        print(f"   Position: {btc_quantity:.8f} BTC")
        print(f"   Collateral: ${safe_amount:.2f} USDC")
        print("-" * 30)
        
        # Place the futures order using the new futures method
        order_result = await client.market_buy_btc_futures(
            safe_amount,
            reason="IMMEDIATE Nanpin Futures Order Test - User Requested"
        )
        
        if order_result:
            print("✅ FUTURES ORDER EXECUTED SUCCESSFULLY!")
            print(f"   Order ID: {order_result.get('id', 'N/A')}")
            print(f"   Status: {order_result.get('status', 'N/A')}")
            
            if 'quantity' in order_result:
                btc_qty = float(order_result['quantity'])
                print(f"   BTC Quantity: {btc_qty:.8f}")
            
            if 'fillPrice' in order_result:
                fill_price = float(order_result['fillPrice'])
                print(f"   Fill Price: ${fill_price:,.2f}")
            
            if 'quoteQuantity' in order_result:
                quote_qty = float(order_result['quoteQuantity'])
                print(f"   USDC Used: ${quote_qty:.2f}")
            
            print("\n🎯 ORDER PLACEMENT COMPLETE!")
            print("📋 POSITION OPENED:")
            print(f"   ✅ BTC Futures Position: {btc_quantity:.8f} BTC")
            print(f"   ✅ Collateral Used: ${safe_amount:.2f} USDC")
            print("\n💡 TO CLOSE POSITION:")
            print("   1. Go to Backpack Exchange")
            print("   2. Navigate to Futures -> Positions")
            print("   3. Close the BTC position manually")
            print(f"   4. Or place a sell order for {btc_quantity:.8f} BTC")
            
        else:
            print("❌ FUTURES ORDER FAILED!")
            print("   Check logs for detailed error information")
            
    except Exception as e:
        print(f"❌ Order execution failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close client connections
        if hasattr(client, 'session') and client.session:
            await client.session.close()

if __name__ == "__main__":
    asyncio.run(execute_real_futures_order())