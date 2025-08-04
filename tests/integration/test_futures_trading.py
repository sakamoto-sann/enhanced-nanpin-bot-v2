#!/usr/bin/env python3
"""
🎯 Proper Futures Trading Test for Nanpin Bot
Test BTC perpetual futures trading using collateral system
"""

import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exchanges.backpack_client_fixed import BackpackNanpinClient

async def test_futures_trading():
    """Test futures trading functionality properly"""
    
    print("🎯 NANPIN BOT FUTURES TRADING TEST")
    print("=" * 50)
    print("Testing BTC perpetual futures trading with collateral")
    print("This bot accumulates BTC positions, never sells")
    print()
    
    # Initialize client
    api_key = os.getenv('BACKPACK_API_KEY')
    secret_key = os.getenv('BACKPACK_SECRET_KEY')
    client = BackpackNanpinClient(api_key, secret_key)
    
    try:
        print("1️⃣ CONNECTION & AUTHENTICATION TEST")
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
        
        print("\n2️⃣ ACCOUNT & COLLATERAL ANALYSIS")
        print("-" * 40)
        
        # Get collateral info for futures trading
        collateral_info = await client.get_collateral_info()
        if collateral_info:
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
            if net_equity_available < 10:
                print("❌ Insufficient collateral for futures trading")
                return
                
        else:
            print("❌ Could not get collateral information")
            return
        
        print("\n3️⃣ CURRENT POSITION STATUS")
        print("-" * 40)
        
        # Check existing BTC futures position
        try:
            positions = await client._make_request(
                'GET',
                '/api/v1/positions',
                signed=True,
                instruction='positionQuery'
            )
            
            btc_position = None
            if positions:
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
        
        print("\n4️⃣ MARKET DATA VALIDATION")
        print("-" * 40)
        
        # Get BTC price for futures
        btc_price = await client.get_btc_price()
        print(f"   BTC Price: ${btc_price:,.2f}")
        
        # Get mark price (futures-specific)
        try:
            mark_price = await client.get_mark_price('BTC_USDC')
            if mark_price:
                print(f"   Mark Price: ${mark_price:,.2f}")
            else:
                print("   Mark Price: Using BTC price as fallback")
        except:
            print("   Mark Price: Not available")
        
        print("\n5️⃣ ORDER PLACEMENT TEST")
        print("-" * 40)
        
        # Test order sizing
        test_amount = 10.0  # $10 USDC minimum for futures
        safe_amount = await client.calculate_safe_order_size(test_amount)
        
        print(f"   Requested Order Size: ${test_amount:.2f}")
        print(f"   Safe Order Size: ${safe_amount:.2f}")
        
        if safe_amount <= 0:
            print("❌ Order rejected by safety checks")
            return
        
        print(f"   ⚠️  This will open a BTC futures position!")
        print(f"   ⚠️  Position will accumulate (bot never sells)")
        
        # Get user confirmation for real futures trading
        try:
            confirm = input("\nType 'FUTURES' to place real futures order: ")
            if confirm != 'FUTURES':
                print("❌ Test cancelled by user")
                return
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Test cancelled by user")
            return
        
        print("\n🚀 PLACING FUTURES ORDER...")
        print("-" * 30)
        
        # Place the futures order
        order_result = await client.market_buy_btc(
            safe_amount,
            reason="Nanpin Futures Test"
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
            
            print("\n⏳ Waiting for position update...")
            await asyncio.sleep(5)
            
            # Check updated position
            try:
                updated_positions = await client._make_request(
                    'GET',
                    '/api/v1/positions',
                    signed=True,
                    instruction='positionQuery'
                )
                
                if updated_positions:
                    for pos in updated_positions:
                        if 'BTC' in pos.get('symbol', ''):
                            new_size = float(pos.get('size', 0))
                            print(f"✅ Updated BTC Position: {new_size:.8f} BTC")
                            break
                
            except Exception as e:
                print(f"   Position update check error: {e}")
            
            print("\n🎉 NANPIN BOT FUTURES TRADING: SUCCESSFUL!")
            print("✅ Bot can place futures orders using collateral")
            print("✅ Position accumulation working")
            print("✅ Ready for live nanpin/DCA strategy")
            
        else:
            print("❌ FUTURES ORDER FAILED")
            print("   Check API permissions for futures trading")
            print("   Verify collateral requirements")
            print("   Ensure futures trading is enabled on account")
    
    except Exception as e:
        print(f"❌ Test error: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Provide specific guidance based on error
        error_str = str(e).lower()
        if 'insufficient' in error_str:
            print("\n💡 SOLUTION:")
            print("   • Your funds are in lending positions")
            print("   • Backpack futures requires actual USDC balance")
            print("   • Withdraw some USDC from lending to spot wallet")
            print("   • Then use spot balance as collateral for futures")
            
        elif 'permission' in error_str or 'unauthorized' in error_str:
            print("\n💡 SOLUTION:")
            print("   • Enable futures trading in Backpack settings")
            print("   • Check API key has futures trading permissions")
            print("   • Verify account is approved for margin/futures")
    
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_futures_trading())