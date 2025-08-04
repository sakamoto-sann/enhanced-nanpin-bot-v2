#!/usr/bin/env python3
"""
🔍 Verify if orders were actually placed on Backpack Exchange
Check order history and positions to confirm real trading activity
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

async def verify_real_orders():
    """Check if our test orders were actually placed"""
    print("🔍 VERIFYING REAL ORDER ACTIVITY")
    print("=" * 50)
    print("Checking Backpack Exchange for actual order history...")
    print()
    
    # Initialize client
    api_key = os.getenv('BACKPACK_API_KEY')
    secret_key = os.getenv('BACKPACK_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ API credentials not found")
        return
    
    client = BackpackNanpinClient(api_key, secret_key)
    
    try:
        # Check recent order history
        print("📋 RECENT ORDER HISTORY")
        print("-" * 30)
        
        try:
            orders = await client._make_request(
                'GET',
                '/api/v1/history/orders',
                {'limit': 10},
                signed=True,
                instruction='orderHistoryQuery'
            )
            
            if orders and len(orders) > 0:
                print(f"Found {len(orders)} recent orders:")
                
                for i, order in enumerate(orders[:10]):
                    order_id = order.get('id', 'N/A')
                    symbol = order.get('symbol', 'N/A')
                    side = order.get('side', 'N/A')
                    status = order.get('status', 'N/A')
                    timestamp = order.get('timestamp', 'N/A')
                    quantity = order.get('quantity', 'N/A')
                    
                    print(f"\n{i+1}. Order ID: {order_id}")
                    print(f"   Symbol: {symbol}")
                    print(f"   Side: {side}")
                    print(f"   Status: {status}")
                    print(f"   Quantity: {quantity}")
                    print(f"   Time: {timestamp}")
                    
                    # Check if this matches our test order
                    if order_id == '4393065718':
                        print("   🎯 THIS IS OUR TEST ORDER!")
                        
            else:
                print("❌ No recent orders found in history")
                
        except Exception as e:
            print(f"❌ Error getting order history: {e}")
        
        # Check current positions
        print("\n📊 CURRENT POSITIONS")
        print("-" * 25)
        
        try:
            positions = await client._make_request(
                'GET',
                '/api/v1/positions',
                signed=True,
                instruction='positionQuery'
            )
            
            if positions and len(positions) > 0:
                print(f"Found {len(positions)} positions:")
                
                total_positions = 0
                for pos in positions:
                    symbol = pos.get('symbol', 'Unknown')
                    size = float(pos.get('size', 0))
                    entry_price = float(pos.get('entryPrice', 0))
                    mark_price = float(pos.get('markPrice', 0))
                    
                    if abs(size) > 0.00000001:  # Non-zero position
                        total_positions += 1
                        print(f"\n   {symbol}:")
                        print(f"   Size: {size:.8f}")
                        print(f"   Entry Price: ${entry_price:,.2f}")
                        print(f"   Mark Price: ${mark_price:,.2f}")
                        
                        if 'BTC' in symbol:
                            print("   🎯 THIS IS A BTC POSITION!")
                
                if total_positions == 0:
                    print("ℹ️  All positions have zero size")
                    
            else:
                print("ℹ️  No positions found")
                
        except Exception as e:
            print(f"❌ Error getting positions: {e}")
        
        # Specifically check the order ID from our test
        print("\n🔍 CHECKING SPECIFIC TEST ORDER")
        print("-" * 35)
        
        test_order_id = '4393065718'
        print(f"Looking for Order ID: {test_order_id}")
        
        try:
            specific_order = await client.get_order_status(test_order_id)
            if specific_order:
                print("✅ TEST ORDER FOUND!")
                print(f"   Details: {specific_order}")
            else:
                print("❌ Test order not found in system")
                print("   This suggests the order may not have been actually placed")
                
        except Exception as e:
            print(f"❌ Error checking specific order: {e}")
        
        # Check recent fills/trades
        print("\n📈 RECENT FILLS/TRADES")
        print("-" * 25)
        
        try:
            fills = await client._make_request(
                'GET',
                '/api/v1/history/fills',
                {'limit': 10},
                signed=True,
                instruction='fillHistoryQuery'
            )
            
            if fills and len(fills) > 0:
                print(f"Found {len(fills)} recent fills:")
                
                for i, fill in enumerate(fills[:5]):
                    print(f"\n{i+1}. Fill ID: {fill.get('id', 'N/A')}")
                    print(f"   Symbol: {fill.get('symbol', 'N/A')}")
                    print(f"   Side: {fill.get('side', 'N/A')}")
                    print(f"   Quantity: {fill.get('quantity', 'N/A')}")
                    print(f"   Price: ${float(fill.get('price', 0)):,.2f}")
                    print(f"   Time: {fill.get('timestamp', 'N/A')}")
                    
            else:
                print("❌ No recent fills found")
                
        except Exception as e:
            print(f"❌ Error getting fills: {e}")
        
        # Final assessment
        print("\n🏆 FINAL ASSESSMENT")
        print("=" * 25)
        
        print("Based on the API responses above:")
        print("• Check if Order ID 4393065718 appears in order history")
        print("• Check if any BTC positions exist")
        print("• Check if any recent fills match our test")
        print()
        print("If NONE of these show our test activity,")
        print("then the order was likely simulated, not actually placed.")
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(verify_real_orders())