"""
Ed25519 compatibility layer for Backpack API
Uses cryptography library's Ed25519 implementation instead of the old ed25519 package
"""

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
import base64

class Ed25519PrivateKey:
    """Compatible wrapper for Ed25519 private key operations"""
    
    def __init__(self, private_key_bytes):
        """Initialize with private key bytes"""
        if len(private_key_bytes) == 32:
            # Raw private key
            self._private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
        else:
            raise ValueError("Private key must be 32 bytes")
    
    def sign(self, message):
        """Sign a message"""
        if isinstance(message, str):
            message = message.encode('utf-8')
        return self._private_key.sign(message)
    
    def get_verifying_key(self):
        """Get the public key for verification"""
        return Ed25519PublicKey(self._private_key.public_key())

class Ed25519PublicKey:
    """Compatible wrapper for Ed25519 public key operations"""
    
    def __init__(self, public_key):
        """Initialize with cryptography public key object"""
        self._public_key = public_key
    
    def verify(self, signature, message):
        """Verify a signature"""
        if isinstance(message, str):
            message = message.encode('utf-8')
        try:
            self._public_key.verify(signature, message)
            return True
        except Exception:
            return False
    
    def encode(self, encoding='raw'):
        """Encode the public key"""
        if encoding == 'raw':
            return self._public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
        else:
            raise ValueError("Only 'raw' encoding is supported")

def SigningKey(private_key_bytes):
    """Factory function compatible with old ed25519 API"""
    return Ed25519PrivateKey(private_key_bytes)

def VerifyingKey(public_key_bytes):
    """Factory function compatible with old ed25519 API"""
    public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
    return Ed25519PublicKey(public_key)