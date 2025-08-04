#!/usr/bin/env python3
"""
🔍 Debug Trading Power Calculation
See exactly what's happening with trading power vs order execution
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exchanges.backpack_client_fixed import BackpackNanpinClient

async def debug_trading_power():
    """Debug the trading power calculation step by step"""
    
    print("🔍 DEBUGGING TRADING POWER CALCULATION")
    print("=" * 50)
    
    # Initialize client
    api_key = os.getenv('BACKPACK_API_KEY')
    secret_key = os.getenv('BACKPACK_SECRET_KEY')
    client = BackpackNanpinClient(api_key, secret_key)
    
    try:
        print("1️⃣ Getting collateral info...")
        collateral_info = await client.get_collateral_info()
        if collateral_info:
            net_equity_available = float(collateral_info.get('netEquityAvailable', 0))
            available_balance = float(collateral_info.get('availableBalance', 0))
            
            print(f"   Net Equity Available: ${net_equity_available:.2f}")
            print(f"   Available Balance: ${available_balance:.2f}")
            
            trading_power = max(net_equity_available, available_balance)
            print(f"   Trading Power: ${trading_power:.2f}")
            
            max_safe_size = trading_power * 0.8
            print(f"   Max Safe Size (80%): ${max_safe_size:.2f}")
        
        print("\n2️⃣ Testing safe order size calculation...")
        test_amount = 5.0
        safe_amount = await client.calculate_safe_order_size(test_amount)
        print(f"   Requested: ${test_amount:.2f}")
        print(f"   Safe Amount: ${safe_amount:.2f}")
        
        if safe_amount <= 0:
            print("   ❌ PROBLEM: Safe amount is 0 or negative!")
            return
        
        print("\n3️⃣ Testing direct order placement...")
        print("   Order would use these parameters:")
        
        btc_price = await client.get_btc_price()
        print(f"   BTC Price: ${btc_price:,.2f}")
        
        order_params = {
            'symbol': 'BTC_USDC',
            'side': 'Bid',
            'orderType': 'Market',
            'quoteQuantity': f"{safe_amount:.2f}",
            'timeInForce': 'IOC'
        }
        print(f"   Order Params: {order_params}")
        
        print("\n4️⃣ The issue might be:")
        print("   • Backpack requires spot USDC balance, not just collateral")
        print("   • Market orders need different parameters")
        print("   • API permissions issue")
        print("   • Order size too small for Backpack's minimums")
        
        print("\n5️⃣ Testing with different order types...")
        # Try different approaches
        
        # Test 1: Check if we need to withdraw from lending first
        print("\n   Test 1: Check lending positions")
        try:
            positions = await client._make_request(
                'GET',
                '/api/v1/borrowLend/positions',
                signed=True,
                instruction='borrowLendPositionQuery'
            )
            
            if positions:
                total_lent_usdc = 0
                for pos in positions:
                    if pos.get('symbol') == 'USDC' and pos.get('side') == 'lend':
                        lent_amount = float(pos.get('size', 0))
                        total_lent_usdc += lent_amount
                        print(f"      USDC Lent: ${lent_amount:.2f}")
                
                if total_lent_usdc > 0:
                    print(f"   💡 SOLUTION: You need to withdraw ${total_lent_usdc:.2f} from lending")
                    print("      Go to Backpack web app > Lend > Withdraw USDC to spot wallet")
                    
        except Exception as e:
            print(f"   Lending check error: {e}")
        
        # Test 2: Try minimum order size per Backpack docs
        print("\n   Test 2: Check Backpack minimum order sizes")
        print("      Backpack minimums might be higher than $5")
        print("      Try $10-20 instead")
        
    except Exception as e:
        print(f"❌ Debug error: {e}")
    
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(debug_trading_power())