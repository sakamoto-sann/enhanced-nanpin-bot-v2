#!/usr/bin/env python3
"""
🔐 Test Backpack API Authentication
Direct test of authentication without the bot framework
"""

import asyncio
import aiohttp
import base64
import time
import os
from cryptography.hazmat.primitives.asymmetric import ed25519
from dotenv import load_dotenv

class BackpackAuthTest:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('BACKPACK_API_KEY')
        self.secret_key = os.getenv('BACKPACK_SECRET_KEY')
        self.base_url = "https://api.backpack.exchange"
        
    def _generate_signature(self, instruction: str, params: dict = None):
        """Generate ED25519 signature for Backpack API"""
        # Create timestamp and window
        timestamp = int(time.time() * 1000)
        window = 5000
        
        # Build signature string
        sign_str_parts = [f"instruction={instruction}"]
        
        # Add sorted parameters
        if params:
            for key in sorted(params.keys()):
                value = params[key]
                if isinstance(value, bool):
                    value = str(value).lower()
                sign_str_parts.append(f"{key}={value}")
        
        # Add timestamp and window
        sign_str_parts.append(f"timestamp={timestamp}")
        sign_str_parts.append(f"window={window}")
        
        # Create signing string
        sign_str = "&".join(sign_str_parts)
        print(f"🔐 Signing string: {sign_str}")
        
        # Sign with Ed25519
        private_key_bytes = base64.b64decode(self.secret_key)
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
        signature = private_key.sign(sign_str.encode('utf-8'))
        signature_b64 = base64.b64encode(signature).decode('utf-8')
        
        return signature_b64, str(timestamp), str(window)
    
    async def test_public_endpoint(self):
        """Test a public endpoint first"""
        print("🌐 Testing public endpoint...")
        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.base_url}/api/v1/ticker"
                params = {"symbol": "BTC_USDC"}
                
                async with session.get(url, params=params) as response:
                    print(f"   Status: {response.status}")
                    data = await response.json()
                    print(f"   Response: {data}")
                    
                    if response.status == 200:
                        print("✅ Public API is working")
                        return True
                    else:
                        print("❌ Public API failed")
                        return False
                        
            except Exception as e:
                print(f"❌ Public API error: {e}")
                return False
    
    async def test_authenticated_endpoint(self):
        """Test the balance endpoint with authentication"""
        print("🔐 Testing authenticated endpoint...")
        
        async with aiohttp.ClientSession() as session:
            try:
                # Generate signature for balanceQuery
                signature, timestamp, window = self._generate_signature('balanceQuery')
                
                headers = {
                    'X-API-Key': self.api_key,
                    'X-Signature': signature,
                    'X-Timestamp': timestamp,
                    'X-Window': window,
                    'Content-Type': 'application/json'
                }
                
                print(f"   Headers: {headers}")
                
                url = f"{self.base_url}/api/v1/capital"
                async with session.get(url, headers=headers) as response:
                    print(f"   Status: {response.status}")
                    
                    try:
                        data = await response.json()
                        print(f"   Response: {data}")
                    except:
                        text = await response.text()
                        print(f"   Response text: {text}")
                    
                    if response.status == 200:
                        print("✅ Authentication successful!")
                        return True
                    else:
                        print(f"❌ Authentication failed: {response.status}")
                        return False
                        
            except Exception as e:
                print(f"❌ Authentication test error: {e}")
                return False
    
    async def run_tests(self):
        """Run all authentication tests"""
        print("🧪 BACKPACK API AUTHENTICATION TEST")
        print("=" * 50)
        print(f"API Key: {self.api_key}")
        print(f"Secret Key: {self.secret_key[:10]}...")
        print()
        
        # Test public endpoint first
        public_ok = await self.test_public_endpoint()
        print()
        
        # Test authenticated endpoint
        if public_ok:
            auth_ok = await self.test_authenticated_endpoint()
            print()
            
            if auth_ok:
                print("🎉 ALL TESTS PASSED - API Authentication is working!")
            else:
                print("❌ Authentication test failed")
        else:
            print("❌ Skipping authentication test due to public API failure")

async def main():
    test = BackpackAuthTest()
    await test.run_tests()

if __name__ == "__main__":
    asyncio.run(main())