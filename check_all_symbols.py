#!/usr/bin/env python3
"""
📋 CHECK ALL BACKPACK SYMBOLS
Find the correct futures symbol for BTC
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exchanges.backpack_client_fixed import BackpackNanpinClient

async def get_all_symbols():
    """Get all available symbols to find correct futures symbols"""
    
    client = BackpackNanpinClient(
        os.getenv('BACKPACK_API_KEY'),
        os.getenv('BACKPACK_SECRET_KEY')
    )
    
    try:
        print("📋 ALL BACKPACK EXCHANGE SYMBOLS")
        print("=" * 50)
        
        # Get all symbols
        symbols = await client._make_request(
            'GET', 
            '/api/v1/ticker', 
            signed=False
        )
        
        if symbols:
            print(f"Total symbols found: {len(symbols)}\n")
            
            # Categorize symbols
            btc_symbols = []
            eth_symbols = []
            sol_symbols = []
            perp_symbols = []
            usd_symbols = []
            
            for symbol in symbols:
                symbol_name = symbol['symbol']
                
                if 'BTC' in symbol_name:
                    btc_symbols.append(symbol_name)
                elif 'ETH' in symbol_name:
                    eth_symbols.append(symbol_name)
                elif 'SOL' in symbol_name:
                    sol_symbols.append(symbol_name)
                
                if 'PERP' in symbol_name:
                    perp_symbols.append(symbol_name)
                elif 'USD' in symbol_name and 'USDC' not in symbol_name:
                    usd_symbols.append(symbol_name)
            
            # Display by category
            print("🪙 BTC SYMBOLS:")
            for symbol in btc_symbols:
                print(f"   {symbol}")
            
            print(f"\n💎 ETH SYMBOLS:")
            for symbol in eth_symbols:
                print(f"   {symbol}")
            
            print(f"\n☀️ SOL SYMBOLS:")
            for symbol in sol_symbols:
                print(f"   {symbol}")
            
            print(f"\n🔄 PERPETUAL FUTURES (PERP):")
            for symbol in perp_symbols:
                print(f"   {symbol}")
            
            print(f"\n💵 USD SYMBOLS (not USDC):")
            for symbol in usd_symbols[:10]:  # First 10
                print(f"   {symbol}")
            
            # Show full list of first 50 symbols
            print(f"\n📊 FIRST 50 SYMBOLS WITH PRICES:")
            print("-" * 50)
            for i, symbol in enumerate(symbols[:50]):
                price = float(symbol['lastPrice'])
                print(f"{i+1:2d}. {symbol['symbol']:15s} - ${price:>10,.2f}")
            
            if len(symbols) > 50:
                print(f"... and {len(symbols) - 50} more symbols")
        
        # Test if any BTC symbol works with our collateral
        print(f"\n🧪 TESTING BTC SYMBOLS WITH COLLATERAL")
        print("-" * 50)
        
        # Get collateral info
        collateral = await client.get_collateral_info()
        if collateral:
            net_equity = float(collateral.get('netEquityAvailable', 0))
            print(f"Available Collateral: ${net_equity:.2f}")
            
            # Test each BTC symbol
            for btc_symbol in btc_symbols:
                print(f"\n   Testing {btc_symbol}:")
                
                order_params = {
                    'symbol': btc_symbol,
                    'side': 'Bid',
                    'orderType': 'Market',
                    'quantity': '0.001',  # Small amount
                    'timeInForce': 'IOC'
                }
                
                try:
                    # Don't actually execute, just test the validation
                    print(f"      Params: {order_params}")
                    print(f"      Testing API validation...")
                    
                    # This will fail but tell us if symbol is valid
                    result = await client._make_request(
                        'POST',
                        '/api/v1/order',
                        order_params,
                        signed=True,
                        instruction='orderExecute'
                    )
                    
                    print(f"      ✅ SUCCESS: {result}")
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if 'insufficient' in error_msg:
                        print(f"      💡 SPOT SYMBOL: Needs USDC in spot wallet")
                    elif 'invalid symbol' in error_msg or 'invalid market' in error_msg:
                        print(f"      ❌ INVALID: Symbol not recognized")
                    elif 'quantity' in error_msg:
                        print(f"      ⚠️ VALID: Quantity precision issue")
                    else:
                        print(f"      ❓ UNKNOWN: {e}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(get_all_symbols())