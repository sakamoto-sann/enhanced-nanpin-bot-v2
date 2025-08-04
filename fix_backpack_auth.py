#!/usr/bin/env python3
"""
🔐 Backpack API Authentication Fix
Generate proper ED25519 key pairs for Backpack API
"""

import base64
import os
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

def generate_ed25519_keypair():
    """Generate a new ED25519 keypair for Backpack"""
    # Generate private key
    private_key = ed25519.Ed25519PrivateKey.generate()
    
    # Get private key bytes
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    # Get public key
    public_key = private_key.public_key()
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    
    # Encode in base64
    private_b64 = base64.b64encode(private_bytes).decode('utf-8')
    public_b64 = base64.b64encode(public_bytes).decode('utf-8')
    
    return public_b64, private_b64

def analyze_current_keys():
    """Analyze the current keys in .env file"""
    
    # Load current keys
    api_key = os.getenv('BACKPACK_API_KEY', '')
    secret_key = os.getenv('BACKPACK_SECRET_KEY', '')
    
    print("🔐 CURRENT KEYS ANALYSIS")
    print("=" * 50)
    print(f"API Key: {api_key}")
    print(f"API Key Length: {len(api_key)} chars")
    print(f"Secret Key: {secret_key}")  
    print(f"Secret Key Length: {len(secret_key)} chars")
    print()
    
    # Try to decode and check format
    try:
        api_decoded = base64.b64decode(api_key)
        print(f"✅ API Key decodes to {len(api_decoded)} bytes")
        
        if len(api_decoded) == 32:
            print("✅ API Key is 32 bytes (correct for ED25519 public key)")
        else:
            print(f"❌ API Key is {len(api_decoded)} bytes (should be 32)")
            
    except Exception as e:
        print(f"❌ API Key base64 decode failed: {e}")
    
    try:
        secret_decoded = base64.b64decode(secret_key)
        print(f"✅ Secret Key decodes to {len(secret_decoded)} bytes")
        
        if len(secret_decoded) == 32:
            print("✅ Secret Key is 32 bytes (correct for ED25519 private key)")
        else:
            print(f"❌ Secret Key is {len(secret_decoded)} bytes (should be 32)")
            
    except Exception as e:
        print(f"❌ Secret Key base64 decode failed: {e}")
    
    print()
    
    # Test if secret key can create valid signature
    if len(secret_key) > 0:
        try:
            # Try to create private key from secret
            secret_decoded = base64.b64decode(secret_key)
            if len(secret_decoded) == 32:
                private_key = ed25519.Ed25519PrivateKey.from_private_bytes(secret_decoded)
                public_key = private_key.public_key()
                
                # Get the derived public key
                derived_public_bytes = public_key.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw
                )
                derived_public_b64 = base64.b64encode(derived_public_bytes).decode('utf-8')
                
                print(f"✅ Derived public key from secret: {derived_public_b64}")
                print(f"📊 Current API key:                  {api_key}")
                
                if derived_public_b64 == api_key:
                    print("✅ API Key matches derived public key - KEYS ARE CORRECT!")
                else:
                    print("❌ API Key does NOT match derived public key")
                    print("💡 SOLUTION: Use derived public key as BACKPACK_API_KEY")
                    print(f"   BACKPACK_API_KEY={derived_public_b64}")
                    
        except Exception as e:
            print(f"❌ Key validation failed: {e}")

if __name__ == "__main__":
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    analyze_current_keys()