#!/usr/bin/env python3
"""
💰 Test Futures Margin Balance on Backpack
Check actual available margin for futures trading
"""

import asyncio
import aiohttp
import base64
import time
import os
from cryptography.hazmat.primitives.asymmetric import ed25519
from dotenv import load_dotenv

class BackpackMarginTest:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('BACKPACK_API_KEY')
        self.secret_key = os.getenv('BACKPACK_SECRET_KEY')
        self.base_url = "https://api.backpack.exchange"
        
    def _generate_signature(self, instruction: str, params: dict = None):
        """Generate ED25519 signature for Backpack API"""
        timestamp = int(time.time() * 1000)
        window = 5000
        
        sign_str_parts = [f"instruction={instruction}"]
        
        if params:
            for key in sorted(params.keys()):
                value = params[key]
                if isinstance(value, bool):
                    value = str(value).lower()
                sign_str_parts.append(f"{key}={value}")
        
        sign_str_parts.append(f"timestamp={timestamp}")
        sign_str_parts.append(f"window={window}")
        
        sign_str = "&".join(sign_str_parts)
        
        private_key_bytes = base64.b64decode(self.secret_key)
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
        signature = private_key.sign(sign_str.encode('utf-8'))
        signature_b64 = base64.b64encode(signature).decode('utf-8')
        
        return signature_b64, str(timestamp), str(window)
    
    async def get_collateral(self):
        """Get collateral information"""
        print("💰 Getting collateral information...")
        
        async with aiohttp.ClientSession() as session:
            try:
                signature, timestamp, window = self._generate_signature('collateralQuery')
                
                headers = {
                    'X-API-Key': self.api_key,
                    'X-Signature': signature,
                    'X-Timestamp': timestamp,
                    'X-Window': window,
                    'Content-Type': 'application/json'
                }
                
                url = f"{self.base_url}/api/v1/capital/collateral"
                async with session.get(url, headers=headers) as response:
                    print(f"   Status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Collateral: {data}")
                        return data
                    else:
                        text = await response.text()
                        print(f"❌ Error: {text}")
                        return None
                        
            except Exception as e:
                print(f"❌ Collateral error: {e}")
                return None
    
    async def check_futures_eligibility(self):
        """Check if account can trade futures"""
        print("🚀 Checking futures trading eligibility...")
        
        # Check account settings
        async with aiohttp.ClientSession() as session:
            try:
                signature, timestamp, window = self._generate_signature('accountQuery')
                
                headers = {
                    'X-API-Key': self.api_key,
                    'X-Signature': signature,
                    'X-Timestamp': timestamp,
                    'X-Window': window,
                    'Content-Type': 'application/json'
                }
                
                url = f"{self.base_url}/api/v1/account"
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        leverage_limit = data.get('leverageLimit', 0)
                        position_limit = data.get('positionLimit', 0)
                        liquidating = data.get('liquidating', False)
                        
                        print(f"   Leverage Limit: {leverage_limit}x")
                        print(f"   Position Limit: ${position_limit:,}")
                        print(f"   Liquidating: {liquidating}")
                        
                        if int(leverage_limit) > 1 and not liquidating:
                            print("✅ Account is eligible for futures trading")
                            return True
                        else:
                            print("❌ Account is NOT eligible for futures trading")
                            return False
                    else:
                        print(f"❌ Error getting account info: {response.status}")
                        return False
                        
            except Exception as e:
                print(f"❌ Account check error: {e}")
                return False
    
    async def calculate_buying_power(self):
        """Calculate available buying power for futures"""
        print("💪 Calculating futures buying power...")
        
        # Get balance data
        balances = None
        collateral = None
        
        # Get spot balances
        async with aiohttp.ClientSession() as session:
            try:
                signature, timestamp, window = self._generate_signature('balanceQuery')
                
                headers = {
                    'X-API-Key': self.api_key,
                    'X-Signature': signature,
                    'X-Timestamp': timestamp,
                    'X-Window': window,
                    'Content-Type': 'application/json'
                }
                
                url = f"{self.base_url}/api/v1/capital"
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        balances = await response.json()
                        print(f"   Spot Balances: {balances}")
                        
            except Exception as e:
                print(f"   Balance error: {e}")
        
        # Get collateral
        collateral = await self.get_collateral()
        
        # Calculate buying power
        total_usdc_value = 0
        
        if balances:
            # Calculate USDC equivalent of all holdings
            usdc_balance = float(balances.get('USDC', {}).get('available', 0))
            total_usdc_value += usdc_balance
            
            # For other assets, we'd need to convert to USDC value
            # For now, just show USDC
            print(f"   USDC Available: ${usdc_balance}")
            
            # Get account leverage limit
            leverage_limit = 8  # From previous account check
            
            max_position_value = total_usdc_value * leverage_limit
            print(f"   Max Position Value: ${max_position_value} (with {leverage_limit}x leverage)")
            
            if total_usdc_value > 0:
                print(f"✅ Buying power available: ${total_usdc_value} margin, ${max_position_value} position")
                return total_usdc_value, max_position_value
            else:
                print("❌ No USDC margin available for futures trading")
                return 0, 0
        
        return 0, 0
    
    async def run_tests(self):
        """Run all margin tests"""
        print("💰 FUTURES MARGIN TEST")
        print("=" * 50)
        
        # Check eligibility
        eligible = await self.check_futures_eligibility()
        print()
        
        # Get collateral
        collateral = await self.get_collateral()
        print()
        
        # Calculate buying power
        margin, position_value = await self.calculate_buying_power()
        print()
        
        print("📊 SUMMARY:")
        print(f"   Futures Eligible: {'✅ Yes' if eligible else '❌ No'}")
        print(f"   Available Margin: ${margin}")
        print(f"   Max Position Value: ${position_value}")
        
        if margin > 0:
            print("🎉 Ready for futures trading!")
        else:
            print("💰 Need to deposit USDC or transfer from spot to enable futures trading")

async def main():
    test = BackpackMarginTest()
    await test.run_tests()

if __name__ == "__main__":
    asyncio.run(main())