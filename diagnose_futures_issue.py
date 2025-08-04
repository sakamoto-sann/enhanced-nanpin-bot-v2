#!/usr/bin/env python3
"""
🔍 DIAGNOSE FUTURES TRADING ISSUE
Figure out why futures orders fail despite having collateral
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exchanges.backpack_client_fixed import BackpackNanpinClient

async def diagnose_futures_issue():
    """Deep diagnosis of futures trading issue"""
    
    print("🔍 DIAGNOSING FUTURES TRADING ISSUE")
    print("=" * 50)
    
    client = BackpackNanpinClient(
        os.getenv('BACKPACK_API_KEY'),
        os.getenv('BACKPACK_SECRET_KEY')
    )
    
    try:
        # 1. Check account type and permissions
        print("1️⃣ ACCOUNT TYPE & PERMISSIONS CHECK")
        print("-" * 30)
        
        # Get detailed collateral info
        collateral = await client.get_collateral_info()
        if collateral:
            print("Detailed collateral breakdown:")
            for key, value in collateral.items():
                print(f"   {key}: {value}")
        
        # 2. Compare our order with the successful test order
        print("\n2️⃣ COMPARING ORDER FORMATS")
        print("-" * 30)
        
        # The successful test order from earlier
        print("✅ SUCCESSFUL ORDER (from test_btc_futures_symbol.py):")
        successful_params = {
            'symbol': 'BTC_USDC_PERP',
            'side': 'Bid',
            'orderType': 'Market',
            'quantity': '0.001',
            'timeInForce': 'IOC'
        }
        print(f"   {successful_params}")
        
        # The failing order from nanpin bot
        print("\n❌ FAILING ORDER (from nanpin bot):")
        # Let's see what the bot is actually sending
        btc_price = await client.get_btc_price()
        
        # Simulate bot's order parameters
        usdc_amount = 10.0
        bot_quantity = usdc_amount / btc_price
        
        bot_params = {
            'symbol': 'BTC_USDC_PERP',
            'side': 'Bid',
            'orderType': 'Market',
            'quoteQuantity': f'{usdc_amount:.2f}',  # Bot uses quoteQuantity
            'timeInForce': 'IOC'
        }
        print(f"   {bot_params}")
        
        print("\n💡 KEY DIFFERENCES:")
        print("   Successful: Uses 'quantity' parameter")
        print("   Bot:        Uses 'quoteQuantity' parameter")
        print("   This might be the issue!")
        
        # 3. Test both parameter formats
        print("\n3️⃣ TESTING BOTH PARAMETER FORMATS")
        print("-" * 30)
        
        # Test A: quoteQuantity format (what bot uses)
        print("Test A: quoteQuantity format (bot's current method)")
        try:
            result = await client._make_request(
                'POST',
                '/api/v1/order',
                {
                    'symbol': 'BTC_USDC_PERP',
                    'side': 'Bid',
                    'orderType': 'Market',
                    'quoteQuantity': '10.00',
                    'timeInForce': 'IOC'
                },
                signed=True,
                instruction='orderExecute'
            )
            print(f"   ✅ quoteQuantity works: {result}")
            
        except Exception as e:
            print(f"   ❌ quoteQuantity failed: {e}")
        
        # Test B: quantity format (what worked before)  
        print("\nTest B: quantity format (successful method)")
        try:
            result = await client._make_request(
                'POST',
                '/api/v1/order',
                {
                    'symbol': 'BTC_USDC_PERP',
                    'side': 'Bid',
                    'orderType': 'Market',
                    'quantity': '0.001',
                    'timeInForce': 'IOC'
                },
                signed=True,
                instruction='orderExecute'
            )
            print(f"   ✅ quantity works: {result}")
            
            # Close position immediately
            if result:
                print("   🛑 Closing position...")
                try:
                    close_result = await client._make_request(
                        'POST',
                        '/api/v1/order',
                        {
                            'symbol': 'BTC_USDC_PERP',
                            'side': 'Ask',
                            'orderType': 'Market',
                            'quantity': '0.001',
                            'timeInForce': 'IOC'
                        },
                        signed=True,
                        instruction='orderExecute'
                    )
                    print(f"   ✅ Position closed: {close_result}")
                except Exception as e:
                    print(f"   ⚠️ Close error: {e}")
            
        except Exception as e:
            print(f"   ❌ quantity failed: {e}")
        
        # 4. Check if account needs futures trading enabled
        print("\n4️⃣ ACCOUNT CONFIGURATION CHECK")
        print("-" * 30)
        
        # Try to access margin/leverage settings
        endpoints_to_check = [
            ('/api/v1/account', 'Account info'),
            ('/api/v1/margin', 'Margin settings'),
            ('/api/v1/leverage', 'Leverage settings'),
            ('/api/v1/futuresAccountBalance', 'Futures balance'),
            ('/api/v1/futuresAccount', 'Futures account')
        ]
        
        for endpoint, description in endpoints_to_check:
            try:
                result = await client._make_request(
                    'GET',
                    endpoint,
                    signed=True,
                    instruction='balanceQuery'
                )
                print(f"   ✅ {description}: {result}")
            except Exception as e:
                if '404' in str(e):
                    print(f"   ❌ {description}: Endpoint not found")
                elif '403' in str(e) or 'forbidden' in str(e).lower():
                    print(f"   🔒 {description}: Access forbidden (may need enabling)")
                else:
                    print(f"   ❓ {description}: {e}")
        
        # 5. Final diagnosis and solution
        print("\n5️⃣ DIAGNOSIS & SOLUTION")
        print("-" * 30)
        
        net_equity = float(collateral.get('netEquityAvailable', 0)) if collateral else 0
        
        if net_equity > 20:
            print("✅ SUFFICIENT COLLATERAL AVAILABLE")
            print("❌ BUT ORDERS STILL FAIL")
            print()
            print("💡 LIKELY CAUSES:")
            print("1. quoteQuantity parameter not supported for futures")
            print("2. Account needs futures trading enabled in settings")
            print("3. API key needs futures trading permissions")
            print("4. Different margin account configuration needed")
            print()
            print("🔧 SOLUTIONS TO TRY:")
            print("1. Update bot to use 'quantity' instead of 'quoteQuantity'")
            print("2. Enable futures trading in Backpack account settings")
            print("3. Check API key has futures permissions")
            print("4. Try portfolio margin mode if available")
        
    except Exception as e:
        print(f"❌ Diagnosis error: {e}")
    
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(diagnose_futures_issue())