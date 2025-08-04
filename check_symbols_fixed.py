#!/usr/bin/env python3
"""
📋 CHECK BACKPACK SYMBOLS - FIXED VERSION
Find the correct symbols without parameters
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exchanges.backpack_client_fixed import BackpackNanpinClient

async def check_symbols():
    """Check symbols and determine if BTC_USDC is spot or futures"""
    
    client = BackpackNanpinClient(
        os.getenv('BACKPACK_API_KEY'),
        os.getenv('BACKPACK_SECRET_KEY')
    )
    
    try:
        print("📋 BACKPACK SYMBOL ANALYSIS")
        print("=" * 50)
        
        # Try different ticker endpoints
        print("1️⃣ TESTING TICKER ENDPOINTS")
        print("-" * 30)
        
        # Test ticker without parameters (should get all)
        try:
            all_tickers = await client._make_request(
                'GET', 
                '/api/v1/ticker',
                {},  # Empty params
                signed=False
            )
            
            if all_tickers and isinstance(all_tickers, list):
                print(f"✅ Found {len(all_tickers)} symbols")
                
                # Extract BTC symbols
                btc_symbols = [t['symbol'] for t in all_tickers if 'BTC' in t['symbol']]
                print(f"BTC symbols: {btc_symbols}")
                
                # Check if BTC_USDC exists
                if 'BTC_USDC' in btc_symbols:
                    print("✅ BTC_USDC found in symbol list")
                else:
                    print("❌ BTC_USDC not found")
                
                # Look for perpetual patterns
                perp_symbols = [t['symbol'] for t in all_tickers if 'PERP' in t['symbol'] or '_USD' in t['symbol']]
                print(f"Perpetual-like symbols: {perp_symbols[:10]}")
                
            else:
                print("❌ No ticker data received")
                
        except Exception as e:
            print(f"Ticker error: {e}")
        
        # 2. Test specific BTC_USDC ticker
        print("\n2️⃣ TESTING BTC_USDC SPECIFICALLY")
        print("-" * 30)
        
        try:
            btc_ticker = await client._make_request(
                'GET',
                '/api/v1/ticker',
                {'symbol': 'BTC_USDC'},
                signed=False
            )
            
            if btc_ticker:
                print(f"✅ BTC_USDC ticker: {btc_ticker}")
                
                # Check if this looks like futures or spot
                if 'markPrice' in btc_ticker or 'fundingRate' in btc_ticker:
                    print("💡 BTC_USDC appears to be FUTURES (has mark price/funding)")
                else:
                    print("💡 BTC_USDC appears to be SPOT (no futures-specific fields)")
            
        except Exception as e:
            print(f"BTC_USDC ticker error: {e}")
        
        # 3. Check account type and configuration
        print("\n3️⃣ ACCOUNT CONFIGURATION CHECK")
        print("-" * 30)
        
        # Check balances (spot)
        try:
            balances = await client._make_request(
                'GET',
                '/api/v1/capital',
                signed=True,
                instruction='balanceQuery'
            )
            
            if balances:
                print("✅ Spot balances accessible")
                usdc_balance = balances.get('USDC', {}).get('available', '0')
                print(f"   USDC spot balance: {usdc_balance}")
        
        except Exception as e:
            print(f"Balance check error: {e}")
        
        # Check positions (futures)
        try:
            positions = await client._make_request(
                'GET',
                '/api/v1/position',
                signed=True,
                instruction='positionQuery'
            )
            
            if positions is not None:
                print("✅ Futures positions accessible")
                print(f"   Current positions: {len(positions) if positions else 0}")
        
        except Exception as e:
            print(f"Position check error: {e}")
        
        # 4. Test small order to see exact error
        print("\n4️⃣ DIAGNOSIS: TESTING ORDER TYPES")
        print("-" * 30)
        
        collateral = await client.get_collateral_info()
        if collateral:
            net_equity = float(collateral.get('netEquityAvailable', 0))
            print(f"Available collateral: ${net_equity:.2f}")
            
            # Test 1: Market order with quoteQuantity (spot-like)
            print("\n   Test A: Market order with quoteQuantity (SPOT style)")
            spot_params = {
                'symbol': 'BTC_USDC',
                'side': 'Bid',
                'orderType': 'Market',
                'quoteQuantity': '10.00',  # $10 worth
                'timeInForce': 'IOC'
            }
            
            try:
                result = await client._make_request(
                    'POST',
                    '/api/v1/order',
                    spot_params,
                    signed=True,
                    instruction='orderExecute'
                )
                print(f"      ✅ SPOT-style order worked: {result}")
                
            except Exception as e:
                error_msg = str(e).lower()
                if 'insufficient funds' in error_msg:
                    print("      💡 SPOT TRADING: Needs USDC in spot wallet")
                    print("      🔍 CONCLUSION: BTC_USDC is SPOT trading pair")
                else:
                    print(f"      ❓ Error: {e}")
            
            # Test 2: Market order with quantity (futures-like)
            print("\n   Test B: Market order with quantity (FUTURES style)")
            btc_price = await client.get_btc_price()
            futures_params = {
                'symbol': 'BTC_USDC',
                'side': 'Bid',
                'orderType': 'Market',
                'quantity': f'{10.0 / btc_price:.6f}',  # $10 worth of BTC
                'timeInForce': 'IOC'
            }
            
            try:
                result = await client._make_request(
                    'POST',
                    '/api/v1/order',
                    futures_params,
                    signed=True,
                    instruction='orderExecute'
                )
                print(f"      ✅ FUTURES-style order worked: {result}")
                
            except Exception as e:
                error_msg = str(e).lower()
                if 'insufficient funds' in error_msg:
                    print("      💡 Still insufficient funds with quantity")
                elif 'quantity decimal' in error_msg:
                    print("      ⚠️ Quantity precision issue")
                else:  
                    print(f"      ❓ Error: {e}")
        
        # 5. Final determination
        print("\n5️⃣ FINAL ANALYSIS")
        print("-" * 30)
        print("Based on the tests:")
        print("1. If 'insufficient funds' with $75+ collateral → SPOT trading")
        print("2. If position endpoint works but orders fail → Account config issue")
        print("3. SOLUTION: Either enable futures OR move funds to spot wallet")
        
        # Check if there are proper futures symbols
        print("\n💡 RECOMMENDATION:")
        print("1. BTC_USDC appears to be SPOT trading (needs USDC in spot wallet)")
        print("2. For FUTURES with collateral, look for BTC perpetual symbols")
        print("3. Or enable cross-margin/portfolio margin in account settings")
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
    
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(check_symbols())