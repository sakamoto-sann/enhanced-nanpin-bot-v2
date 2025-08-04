#!/usr/bin/env python3
"""
🔍 Backpack Account Balance Analyzer
Find where your $76.06 is located and how to make it available for trading
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exchanges.backpack_client_fixed import BackpackNanpinClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccountAnalyzer:
    """Analyze Backpack account to find where funds are located"""
    
    def __init__(self):
        self.client = None
    
    async def analyze_complete_account(self):
        """Complete account analysis"""
        try:
            print("🔍 BACKPACK ACCOUNT ANALYSIS")
            print("=" * 50)
            print(f"Analysis time: {datetime.now().isoformat()}")
            print()
            
            # Initialize client
            await self._init_client()
            
            # 1. Capital balances
            print("💰 CAPITAL BALANCES (/api/v1/capital)")
            print("-" * 40)
            await self._check_capital_balances()
            
            # 2. Collateral information
            print("\n🏦 COLLATERAL INFORMATION (/api/v1/capital/collateral)")
            print("-" * 50)
            await self._check_collateral_detailed()
            
            # 3. Try borrow-lend positions (with proper implementation)
            print("\n💳 BORROW-LEND POSITIONS")
            print("-" * 30)
            await self._check_borrow_lend_positions()
            
            # 4. Check all positions
            print("\n📊 ALL POSITIONS (/api/v1/positions)")
            print("-" * 35)
            await self._check_all_positions()
            
            # 5. Generate recommendations
            print("\n🎯 RECOMMENDATIONS")
            print("-" * 20)
            await self._generate_recommendations()
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
        finally:
            if self.client:
                await self.client.close()
    
    async def _init_client(self):
        """Initialize client"""
        api_key = os.getenv('BACKPACK_API_KEY')
        secret_key = os.getenv('BACKPACK_SECRET_KEY')
        
        if not api_key or not secret_key:
            raise Exception("Missing API credentials")
        
        self.client = BackpackNanpinClient(api_key, secret_key)
        print("✅ Client initialized")
    
    async def _check_capital_balances(self):
        """Check capital balances in detail"""
        try:
            balances = await self.client.get_balances()
            
            if not balances:
                print("❌ No balance data returned")
                return
            
            print(f"Found {len(balances)} assets:")
            
            total_value = 0
            for asset, data in balances.items():
                available = float(data.get('available', 0))
                locked = float(data.get('locked', 0))
                staked = float(data.get('staked', 0))
                total = available + locked + staked
                
                if total > 0:
                    print(f"  {asset}:")
                    print(f"    Available: {available:.8f}")
                    print(f"    Locked: {locked:.8f}")
                    print(f"    Staked: {staked:.8f}")
                    print(f"    Total: {total:.8f}")
                    
                    if asset == 'USDC':
                        total_value += total
                    elif asset == 'BTC':
                        # Get BTC price for value calculation
                        try:
                            btc_price = await self.client.get_btc_price()
                            btc_value = total * btc_price
                            total_value += btc_value
                            print(f"    USD Value: ${btc_value:.2f} (@ ${btc_price:,.2f})")
                        except:
                            pass
            
            print(f"\n💰 Total Capital Value: ${total_value:.2f}")
            
        except Exception as e:
            print(f"❌ Capital balance error: {e}")
    
    async def _check_collateral_detailed(self):
        """Check detailed collateral information"""
        try:
            collateral = await self.client.get_collateral_info()
            
            if not collateral:
                print("❌ No collateral data returned")
                return
            
            print("Collateral Details:")
            for key, value in collateral.items():
                if value is not None:
                    if isinstance(value, (int, float)):
                        if abs(float(value)) > 0.0001:  # Only show non-zero values
                            print(f"  {key}: {float(value):.8f}")
                    else:
                        print(f"  {key}: {value}")
            
            # Key metrics
            net_equity = float(collateral.get('netEquity', 0))
            available_balance = float(collateral.get('availableBalance', 0))
            margin_used = net_equity - available_balance
            
            print(f"\n📊 Key Metrics:")
            print(f"  Net Equity: ${net_equity:.2f}")
            print(f"  Available Balance: ${available_balance:.2f}")
            print(f"  Margin Used/Locked: ${margin_used:.2f}")
            
            if margin_used > 0:
                print(f"\n⚠️  ${margin_used:.2f} appears to be locked in positions or lending")
            
        except Exception as e:
            print(f"❌ Collateral error: {e}")
    
    async def _check_borrow_lend_positions(self):
        """Check borrow-lend positions using direct API call"""
        try:
            # Try direct API call to borrow-lend positions endpoint
            borrow_lend_positions = await self.client._make_request(
                'GET',
                '/api/v1/borrowLend/positions',
                signed=True,
                instruction='borrowLendPositionQuery'
            )
            
            if not borrow_lend_positions:
                print("ℹ️  No borrow-lend positions found")
                return
            
            print(f"Found {len(borrow_lend_positions)} borrow-lend positions:")
            
            total_lent = 0
            total_borrowed = 0
            
            for position in borrow_lend_positions:
                symbol = position.get('symbol', 'Unknown')
                side = position.get('side', 'Unknown')
                size = float(position.get('size', 0))
                rate = float(position.get('rate', 0))
                
                if size > 0:
                    print(f"  {symbol} ({side.upper()}):")
                    print(f"    Size: {size:.8f}")
                    print(f"    Rate: {rate:.4%}")
                    
                    if side.lower() == 'lend' and symbol == 'USDC':
                        total_lent += size
                        print(f"    💰 This is lent USDC that could be withdrawn!")
                    elif side.lower() == 'borrow' and symbol == 'USDC':
                        total_borrowed += size
            
            if total_lent > 0:
                print(f"\n🎯 FOUND IT! ${total_lent:.2f} USDC is in lending positions")
                print("   You need to withdraw this from lending to make it available for trading")
            
            if total_borrowed > 0:
                print(f"\n📊 You have ${total_borrowed:.2f} USDC borrowed")
            
        except Exception as e:
            print(f"❌ Borrow-lend positions error: {e}")
            print("   This might be the issue - unable to check lending positions")
    
    async def _check_all_positions(self):
        """Check all trading positions"""
        try:
            positions = await self.client._make_request(
                'GET',
                '/api/v1/positions',
                signed=True,
                instruction='positionQuery'
            )
            
            if not positions:
                print("ℹ️  No trading positions found")
                return
            
            print(f"Found {len(positions)} trading positions:")
            
            for position in positions:
                symbol = position.get('symbol', 'Unknown')
                size = float(position.get('size', 0))
                entry_price = float(position.get('entryPrice', 0))
                mark_price = float(position.get('markPrice', 0))
                unrealized_pnl = float(position.get('unrealizedPnl', 0))
                
                if abs(size) > 0.0001:  # Only show positions with size
                    print(f"  {symbol}:")
                    print(f"    Size: {size:.8f}")
                    print(f"    Entry Price: ${entry_price:.2f}")
                    print(f"    Mark Price: ${mark_price:.2f}")
                    print(f"    Unrealized PnL: ${unrealized_pnl:.2f}")
                    
                    # Calculate position value
                    position_value = abs(size) * mark_price
                    print(f"    Position Value: ${position_value:.2f}")
            
        except Exception as e:
            print(f"❌ Positions error: {e}")
    
    async def _generate_recommendations(self):
        """Generate actionable recommendations"""
        try:
            print("Based on the analysis above:")
            print()
            print("1. 📊 Account Status:")
            print("   - Net Equity: $76.06 (you have funds)")
            print("   - Available Balance: $0.00 (locked/lent)")
            print()
            print("2. 🔍 Likely Issues:")
            print("   - USDC may be in lending positions")
            print("   - Funds might be locked as collateral")
            print("   - Could be in unrealized trading positions")
            print()
            print("3. 🎯 Next Steps:")
            print("   - Check Backpack Exchange web interface")
            print("   - Look for 'Lending' or 'Earn' section")
            print("   - Withdraw any lent USDC back to spot wallet")
            print("   - Close any open positions if not needed")
            print()
            print("4. 🛠️  If funds are lent:")
            print("   - Go to Backpack web app > Lend section")
            print("   - Find your USDC lending position")
            print("   - Click 'Withdraw' or 'Stop Lending'")
            print("   - Move funds back to spot wallet")
            print()
            print("5. 🚀 Once available:")
            print("   - Re-run the order placement test")
            print("   - Bot should be able to place orders")
            
        except Exception as e:
            print(f"❌ Recommendation error: {e}")

async def main():
    """Main analysis function"""
    analyzer = AccountAnalyzer()
    await analyzer.analyze_complete_account()

if __name__ == "__main__":
    asyncio.run(main())