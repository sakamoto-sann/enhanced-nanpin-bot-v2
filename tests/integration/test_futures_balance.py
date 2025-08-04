#!/usr/bin/env python3
"""
🚀 Test Futures Account Balance on Backpack
Check futures margin and collateral, not spot USDC
"""

import asyncio
import aiohttp
import base64
import time
import os
from cryptography.hazmat.primitives.asymmetric import ed25519
from dotenv import load_dotenv

class BackpackFuturesTest:
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
    
    async def get_futures_positions(self):
        """Get futures positions"""
        print("🚀 Getting futures positions...")
        
        async with aiohttp.ClientSession() as session:
            try:
                signature, timestamp, window = self._generate_signature('positionQuery')
                
                headers = {
                    'X-API-Key': self.api_key,
                    'X-Signature': signature,
                    'X-Timestamp': timestamp,
                    'X-Window': window,
                    'Content-Type': 'application/json'
                }
                
                url = f"{self.base_url}/api/v1/position"
                async with session.get(url, headers=headers) as response:
                    print(f"   Status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Futures positions: {data}")
                        return data
                    else:
                        text = await response.text()
                        print(f"❌ Error: {text}")
                        return None
                        
            except Exception as e:
                print(f"❌ Futures positions error: {e}")
                return None
    
    async def get_futures_account(self):
        """Try to get futures account information"""
        print("💰 Getting futures account info...")
        
        # Try different possible endpoints
        endpoints = [
            ('/api/v1/capital/futures', 'futuresBalanceQuery'),
            ('/api/v1/account/futures', 'futuresAccountQuery'),
            ('/api/v1/margin', 'marginQuery'),
            ('/api/v1/account', 'accountQuery')
        ]
        
        async with aiohttp.ClientSession() as session:
            for endpoint, instruction in endpoints:
                try:
                    print(f"   Trying {endpoint} with {instruction}...")
                    
                    signature, timestamp, window = self._generate_signature(instruction)
                    
                    headers = {
                        'X-API-Key': self.api_key,
                        'X-Signature': signature,
                        'X-Timestamp': timestamp,
                        'X-Window': window,
                        'Content-Type': 'application/json'
                    }
                    
                    url = f"{self.base_url}{endpoint}"
                    async with session.get(url, headers=headers) as response:
                        print(f"     Status: {response.status}")
                        
                        if response.status == 200:
                            data = await response.json()
                            print(f"   ✅ Success: {data}")
                            return data
                        else:
                            text = await response.text()
                            print(f"     Error: {text}")
                            
                except Exception as e:
                    print(f"     Exception: {e}")
            
            print("❌ No futures account endpoint found")
            return None
    
    async def run_tests(self):
        """Run futures account tests"""
        print("🚀 FUTURES ACCOUNT TEST")
        print("=" * 50)
        
        # Check futures positions
        positions = await self.get_futures_positions()
        print()
        
        # Check futures account
        account = await self.get_futures_account()
        print()
        
        if positions is not None or account is not None:
            print("🎉 Found futures account data!")
        else:
            print("❌ No futures account data found")

async def main():
    test = BackpackFuturesTest()
    await test.run_tests()

if __name__ == "__main__":
    asyncio.run(main())